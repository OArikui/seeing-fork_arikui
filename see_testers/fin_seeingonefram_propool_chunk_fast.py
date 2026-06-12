import os
from time import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import sqlite3
import datetime
import logging
import random
import sys
from PIL import Image  # pip install pillow
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from MIN2_ignore_sunspots import MIN2_ignore_sunspots as MIN2_ver1
from db_for_seeing import save_db

logging.basicConfig(level=logging.INFO)

def _normalize_path(p):
    if isinstance(p, Path):
        p = str(p)
    if isinstance(p, bytes):
        p = p.decode('utf-8', errors='ignore')
    if p.startswith("file://"):
        p = p[7:] if p.startswith("file:///") else p[7:]
    try:
        pp = Path(p).expanduser().resolve()
    except Exception:
        pp = Path(p).expanduser()
    try:
        return pp.as_posix()
    except Exception:
        return str(pp)

def imread_safe(path, flags=cv2.IMREAD_UNCHANGED):
    pnorm = _normalize_path(path)
    try:
        img = cv2.imread(pnorm, flags)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        arr = np.fromfile(pnorm, dtype=np.uint8)
        if arr.size == 0:
            return None
        img = cv2.imdecode(arr, flags)
        return img
    except Exception as e:
        logging.debug(f"imread_safe failed for {path}: {e}")
        return None

def get_file_size(path):
    try:
        return os.stat(path).st_size
    except Exception:
        return 0

#  PIL ヘッダ読みで展開サイズを推定 --
def estimate_uncompressed_bytes_from_pil(path):
    """
    PIL でヘッダだけ読み、幅×高さ×bytes_per_pixel を返す。
    失敗したら None を返す（呼び出し側でフォールバックする）。
    """
    try:
        with Image.open(path) as im:
            w, h = im.size
            mode = im.mode  # 'L','RGB','I;16' など
            # おおよその bytes per pixel を推定
            if '16' in mode or mode.startswith('I;16'):
                bpp = 2
            elif mode == 'F':
                bpp = 4
            elif mode == 'RGB':
                bpp = 3
            elif mode == 'RGBA':
                bpp = 4
            elif mode == 'L' or mode == 'P':
                bpp = 1
            else:
                # 不明なモードは 1 として扱う（保守的）
                bpp = 1
            return int(w * h * bpp)
    except Exception:
        return None

#  標本サンプリングで補正係数を求める 
def compute_bin_correction_ratios(paths, bins, sample_per_bin=20):
    """
    paths: list of Path or str
    bins: list/array of bin edges in bytes (for file size)
    sample_per_bin: 標本数（各ビン）
    戻り値: dict mapping bin_index -> median_ratio (uncmp_bytes / file_size)
    """
    """
    この関数はファイル一覧をビン（bins）で分類し、
    各ビンから最大 sample_per_bin 個を無作為抽出して (推定非圧縮バイト数) ÷ (ファイルサイズ) の比率の中央値を返します。
    結果はビン番号 → 補正係数（float）の辞書です。
    """
    paths = [Path(p) for p in paths]
    sizes = np.array([get_file_size(p) for p in paths])
    bin_indices = np.digitize(sizes, bins)
    bin_ratios = {}

    for b in np.unique(bin_indices):
        idxs = np.where(bin_indices == b)[0]
        if len(idxs) == 0:
            continue
        k = min(sample_per_bin, len(idxs))
        sampled_idxs = random.sample(list(idxs), k)
        ratios = []
        for si in sampled_idxs:
            p = paths[si]
            uncmp = estimate_uncompressed_bytes_from_pil(p)
            if uncmp is None:
                # フォールバック: 実際にデコードして nbytes を取得（遅いが標本のみ）
                try:
                    img = imread_safe(p)
                    if img is not None:
                        uncmp = img.nbytes
                except Exception:
                    uncmp = None
            fs = get_file_size(p)
            if uncmp is not None and fs > 0:
                ratios.append(uncmp / fs)
        if len(ratios) == 0:
            # 標本で何も取れなかったら保守的に ratio=1.0
            median_ratio = 1.0
        else:
            median_ratio = float(np.median(ratios))
        bin_ratios[int(b)] = median_ratio
    return bin_ratios

#  全ファイルに補正を適用して推定展開サイズを返す 
def estimate_uncompressed_bytes_for_all(paths, bins=None, sample_per_bin=20):
    """
    paths: list of Path or str
    bins: bin edges in bytes. デフォルトは自動生成。
    sample_per_bin: 標本数
    戻り値: dict {str(path): estimated_uncompressed_bytes}
    """
    paths = [Path(p) for p in paths]
    n = len(paths)
    sizes = np.array([get_file_size(p) for p in paths])

    # デフォルトのビン（ファイルサイズの分布に応じて自動生成）
    if bins is None:
        # ログスケールでビンを作る（小〜大ファイルを分ける）
        min_s = max(1, sizes.min()) if sizes.size > 0 else 1
        max_s = max(1, sizes.max()) if sizes.size > 0 else 1
        # 例: 5ビン（調整可）
        bins = np.unique(np.logspace(np.log10(min_s), np.log10(max_s + 1), num=6)).astype(np.int64)
        bins = bins.tolist()

    # 標本からビンごとの補正係数を計算
    bin_ratios = compute_bin_correction_ratios(paths, bins, sample_per_bin=sample_per_bin)

    # 推定を適用
    estimates = {}
    for p, fs in zip(paths, sizes):
        # ビンを決める
        b_idx = int(np.digitize(fs, bins))
        ratio = bin_ratios.get(b_idx, 1.0)
        est = fs * ratio
        # 最低でもファイルサイズ以上（保守的）
        est = max(est, fs)
        estimates[str(p)] = est
    return estimates

# 標本ベースでチャンク分割する関数 
def chunk_by_sampled_estimate(paths, max_chunk_bytes=5 * 1024**3, sample_per_bin=20, bins=None):
    """
    paths: list of Path or str
    max_chunk_bytes: チャンクあたりの推定上限バイト
    sample_per_bin: 各ビンの標本数(増やせば)
    bins: ビン境界（bytes）。None なら自動生成
    戻り値: list of chunks (each chunk is list of Path)
    """
    paths = list(paths)
    if len(paths) == 0:
        return []

    # 1) まずファイルサイズだけで簡易分割（高速）して大まかなグループを作る
    sizes = np.array([get_file_size(p) for p in paths])
    # 2) 標本推定で展開サイズを推定
    estimates = estimate_uncompressed_bytes_for_all(paths, bins=bins, sample_per_bin=sample_per_bin)

    # 3) 推定値に基づいてチャンクを作る
    chunks = []
    cur = []
    cur_bytes = 0.0
    for p in paths:
        est = estimates.get(str(p), get_file_size(p))
        if cur and cur_bytes + est > max_chunk_bytes:
            chunks.append(cur)
            cur = [p]
            cur_bytes = est
        else:
            cur.append(p)
            cur_bytes += est
    if cur:
        chunks.append(cur)
    return chunks

from decimal import Decimal,ROUND_HALF_UP
def seeing_one_frame_fast(readed_img, cir_stat, limb_wigth=24, allp_num=1360,secure=False):
    (cx,cy), r = cir_stat
    if allp_num % 4 != 0:
        raise ValueError("allp_numは4の倍数にしてください")

    half = int(allp_num / 4 / 2)
    i_vals = np.arange(-half, half, dtype=np.float64)

    tmp = r * r - i_vals * i_vals
    min2r_arr = np.sqrt(tmp)
    min2rlst = []

    realrlst = []
    if secure:
        h, w = readed_img.shape[:2]

    for idx, min2r in enumerate(min2r_arr):
        min2r_int = float(Decimal(min2r).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        min2rlst.append(float(Decimal(min2r).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)))

        # L
        y = int((cy + i_vals[idx]))
        x_start = int((cx - min2r_int - limb_wigth))
        x_end = int((cx - min2r_int + limb_wigth))
        if secure:
            x_start = max(0, x_start)
            x_end = min(w, x_end)
        samples = readed_img[y, x_start:x_end]
        if samples.size == 0:
            raise ("sample size is zero")
        else:
            grad = np.gradient(samples.astype(np.float64))
            grad_max_idx = np.argmax(np.abs(grad))
            diff = np.diff(grad[:grad_max_idx])
            if diff.size == 0:
                raise ("sample size is zero")
            else:
                realindx = float(np.argmax(diff)) + 0.5#diffで減る分の0.5
        realrlst.append(min2r_int + realindx - limb_wigth)

        # R
        x_start = int((cx + min2r_int - limb_wigth))
        x_end = int((cx + min2r_int + limb_wigth))
        if secure:
            x_start = max(0, x_start)
            x_end = min(w, x_end)
        samples = readed_img[y, x_start:x_end]
        if samples.size == 0:
            raise ("sample size is zero")
        else:
            grad = np.gradient(samples.astype(np.float64))
            grad_max_idx = np.argmax(np.abs(grad))
            diff = np.diff(grad[grad_max_idx:])
            if diff.size == 0:
                raise ("sample size is zero")
            else:
                realindx = float(np.argmax(diff)) + grad_max_idx + 0.5#diffで減る分の0.5
        realrlst.append(min2r_int + realindx - limb_wigth)

        # T
        x = int((cx + i_vals[idx]))
        y_start = int((cy - min2r_int - limb_wigth))
        y_end = int((cy - min2r_int + limb_wigth))
        if secure:
            y_start = max(0, y_start)
            y_end = min(h, y_end)
        samples = readed_img[y_start:y_end, x]
        if samples.size == 0:
            raise ("sample size is zero")
        else:
            grad = np.gradient(samples.astype(np.float64))
            grad_max_idx = np.argmax(np.abs(grad))
            diff = np.diff(grad[:grad_max_idx])
            if diff.size == 0:
                raise ("sample size is zero")
            else:
                realindx = float(np.argmax(diff)) + 0.5#diffで減る分の0.5
        realrlst.append(min2r_int + realindx - limb_wigth)

        # B
        y_start = int((cy + min2r_int - limb_wigth))
        y_end = int((cy + min2r_int + limb_wigth))
        if secure:
            y_start = max(0, y_start)
            y_end = min(h, y_end)
        samples = readed_img[y_start:y_end, x]
        if samples.size == 0:
            raise ("sample size is zero")
        else:
            grad = np.gradient(samples.astype(np.float64))
            grad_max_idx = np.argmax(np.abs(grad))
            diff = np.diff(grad[grad_max_idx:])
            if diff.size == 0:
                raise ("sample size is zero")
            else:
                realindx = float(np.argmax(diff)) + grad_max_idx + 0.5#diffで減る分の0.5
        realrlst.append(min2r_int + realindx - limb_wigth)

    realrlst = np.array(realrlst, dtype=np.float64)
    min2rlst_rep = np.repeat(min2rlst, 4)
    return float(np.std(realrlst - min2rlst_rep))

def process_single_image(img, path, limb_wigth=24, allp_num=1360):
    try:
        if img is None:
            logging.warning(f"Image is None for {path}")
            return path, None

        try:
            gray_for_min2 = ((img >> 8).astype("uint8"))
        except Exception:
            try:
                gray_for_min2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
            except Exception as e:
                logging.exception(f"Gray conversion failed for {path}: {e}")
                return path, None

        try:
            cir_stat = MIN2_ver1(gray_for_min2, n=10, light_threshold=50,
                                 limb_wigth=limb_wigth, show=False, debug=False)
        except Exception as e:
            logging.exception(f"MIN2_ver1 failed for {path}: {e}")
            return path, None

        try:
            seeing_val = seeing_one_frame_fast(img, cir_stat,
                                               limb_wigth=limb_wigth,
                                               allp_num=allp_num)
            return path, seeing_val
        except Exception as e:
            logging.exception(f"seeing_one_frame_fast failed for {path}: {e}")
            return path, None

    except Exception as e:
        logging.exception(f"Unexpected error for {path}: {e}")
        return path, None

def main(peadirpath:str,savepath:str=None,dyjest=False, max_workers=int|None,max_chunk_bytes=5 * 1024**3, sample_per_bin=20,limb_wigth=24, allp_num=1360,debug=False):
    from time import time
    import datetime
    if debug:
        print(f"{__file__.split("\\")[-1]}process start {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}")
        print(f"--seeing_main_Debug--\nselected:{peadirpath}\nmax_workers:{max_workers}")
        st = time()
    peadirpath = Path(peadirpath)

    # collect paths (same as before)
    paths = []
    for sub in peadirpath.iterdir():
        if sub.is_dir():
            paths += [p for p in sub.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff")]

    # chunk by sampled estimate
    chunks = chunk_by_sampled_estimate(paths, max_chunk_bytes=max_chunk_bytes, sample_per_bin=sample_per_bin)

    all_results = {}

    for ci, chunk in enumerate(chunks, start=1):
        print(f"Chunk {ci}/{len(chunks)}: {len(chunk)} files")

        preloaded = []
        for p in tqdm(chunk, desc=f"Loading images (chunk {ci})"):
            img = imread_safe(p, cv2.IMREAD_UNCHANGED)
            if img is not None:
                preloaded.append((p, img))
            else:
                logging.warning(f"Failed to load (skipped): {p}")

        with ProcessPoolExecutor(max_workers=max_workers) as exe:
            futures = {exe.submit(process_single_image, img, p,
                                  limb_wigth, allp_num): p for p, img in preloaded}

            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"Processing images (chunk {ci})"):
                p = futures[fut]
                path, val = fut.result()
                all_results.setdefault(p.parent.name, []).append(val)

        preloaded.clear()

    if debug:
        for group, vals in all_results.items():
            arr = np.array([v for v in vals if v is not None], dtype=float)
            success = arr.size
            fail = len(vals) - success
            logging.info(f"group={group} total={len(vals)} success={success} fail={fail}")
            contents=["mean","min","max","std","median"]
            Df=pd.DataFrame({cdir:pd.Series(all_results[cdir]).agg(contents) for cdir in all_results.keys()})
            print(Df)
        print(f"Total time: {time() - st:.2f} sec")
    if savepath != None:
        save_db(all_results,peadirpath,__file__.split("\\")[-1]+"_prototype",{"limb_wigth":limb_wigth,"secdiff":"diff(grad())"},savepath,dyjest=dyjest)
    return all_results

if __name__ == "__main__":
    from tkinter.filedialog import askdirectory
    peadirpath = askdirectory(title="画像が入っているフォルダを選択してください")
    max_chunk_GB=10
    main(peadirpath, max_workers=None, limb_wigth=24, allp_num=1360,max_chunk_bytes=max_chunk_GB * 1024**3, sample_per_bin=20,debug=True)
