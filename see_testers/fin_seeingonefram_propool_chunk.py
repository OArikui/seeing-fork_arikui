from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm
import sys
parent_dir = str(Path(__file__).resolve().parent.parent)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
from MIN2_ignore_sunspots import MIN2_ignore_sunspots as MIN2_ver1
from db_for_seeing import save_db
import logging
from decimal import Decimal,ROUND_HALF_UP

logging.basicConfig(level=logging.INFO)

def _normalize_path(p):
    """
    Path オブジェクトまたは文字列を正規化して返す。
    - file:// プレフィックスを除去
    - Path.resolve() を使って絶対パス化
    - Windows では as_posix() を使ってスラッシュに変換（cv2 に渡すと安定）
    """
    if isinstance(p, Path):
        p = str(p)
    if isinstance(p, bytes):
        p = p.decode('utf-8', errors='ignore')
    # file:// を取り除く
    if p.startswith("file://"):
        # Windows の file://C:/... などを正しく扱う
        p = p[7:] if p.startswith("file:///") else p[7:]
    # 正規化
    try:
        pp = Path(p).expanduser().resolve()
    except Exception:
        pp = Path(p).expanduser()
    # Windows で長いパスや Unicode の問題がある場合は as_posix() を使う
    try:
        return pp.as_posix()
    except Exception:
        return str(pp)

def imread_safe(path, flags=cv2.IMREAD_UNCHANGED):
    """
    安全な画像読み込み：
    - まず cv2.imread を試す（as_posix で渡す）
    - 失敗したら numpy.fromfile + cv2.imdecode を試す（Windows の Unicode/長いパス対策）
    - 失敗時は None を返す
    """
    pnorm = _normalize_path(path)
    # 1) まず通常読み込み（パスをスラッシュに変換して渡す）
    try:
        img = cv2.imread(pnorm, flags)
        if img is not None:
            return img
    except Exception:
        pass

    # 2) numpy.fromfile + imdecode（バイナリ読み込み）
    try:
        # numpy.fromfile はバイナリを読み込める（Windows の Unicode パスでも動く）
        arr = np.fromfile(pnorm, dtype=np.uint8)
        if arr.size == 0:
            return None
        img = cv2.imdecode(arr, flags)
        return img
    except Exception as e:
        logging.debug(f"imread_safe failed for {path}: {e}")
        return None


# 1枚あたりのメモリ推定（バイト）
def estimate_image_bytes(path):
    img = imread_safe(path)
    if img is None:
        return 0
    return img.nbytes


def seeing_one_frame_fast(readed_img, cir_stat, limb_wigth=24, allp_num=1360):
    (cx,cy),r = cir_stat
    if allp_num % 4 != 0:
        raise ValueError("allp_numは4の倍数にしてください")

    half = int(allp_num / 4 / 2)
    i_vals = np.arange(-half, half, dtype=np.float64)

    tmp = r * r - i_vals * i_vals
    min2r_arr = np.sqrt(tmp)
    min2rlst = []

    realrlst = []
    h, w = readed_img.shape[:2]
    
    for idx, min2r in enumerate(min2r_arr):
        min2r_int = float(Decimal(min2r).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        min2rlst.append(float(Decimal(min2r).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)))
        # L
        y = int((cy + i_vals[idx]))
        x_start = int((cx - min2r_int - limb_wigth))
        x_end = int((cx - min2r_int + limb_wigth))
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
                realindx = float(np.argmax(np.abs(diff))) + 0.5#diffで減る分の0.5
        realrlst.append(min2r_int + realindx - limb_wigth)

        # R
        x_start = int((cx + min2r_int - limb_wigth))
        x_end = int((cx + min2r_int + limb_wigth))
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
                realindx = float(np.argmax(np.abs(diff))) + grad_max_idx + 0.5#diffで減る分の0.5
        realrlst.append(min2r_int + realindx - limb_wigth)

        # T
        x = int((cx + i_vals[idx]))
        y_start = int((cy - min2r_int - limb_wigth))
        y_end = int((cy - min2r_int + limb_wigth))
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
                realindx = float(np.argmax(np.abs(diff))) + 0.5#diffで減る分の0.5
        realrlst.append(min2r_int + realindx - limb_wigth)

        # B
        y_start = int((cy + min2r_int - limb_wigth))
        y_end = int((cy + min2r_int + limb_wigth))
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
                realindx = float(np.argmax(np.abs(diff))) + grad_max_idx + 0.5#diffで減る分の0.5
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



def chunk_paths_by_memory(paths, chunk_bytes=5 * 1024**3):
    chunks = []
    current = []
    current_bytes = 0
    for p in tqdm(paths, desc="Estimating image sizes"):
        b = estimate_image_bytes(p)
        if b == 0:
            continue
        if current_bytes + b > chunk_bytes and current:
            chunks.append(current)
            current = [p]
            current_bytes = b
        else:
            current.append(p)
            current_bytes += b
    if current:
        chunks.append(current)
    return chunks


def main(peadirpath:str, savepath:str=None,dyjest=False,max_workers:int=None, chunk_bytes=5 * 1024**3,limb_wigth=24, allp_num=1360, debug=False):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from time import time
    import pandas as pd
    import datetime
    if debug:
        print(f"{__file__.split("\\")[-1]}process start {datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}")
        st=time()
        print(f"--seeing_main_Debug--\nselected:{peadirpath}\nmax_workers:{max_workers}\nchunk_bytes:{chunk_bytes/1024**3}GB")
    peadirpath = Path(peadirpath)
    paths = []
    for sub in peadirpath.iterdir():
        if sub.is_dir():
            paths += [p for p in sub.iterdir()
                      if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".tiff")]

    chunks = chunk_paths_by_memory(paths, chunk_bytes=chunk_bytes)

    all_results = {}

    for ci, chunk in enumerate(chunks, start=1):
        print(f"Chunk {ci}/{len(chunks)}: {len(chunk)} files")

        preloaded = []
        for p in tqdm(chunk, desc=f"Loading images (chunk {ci})"):
            img = imread_safe(p, cv2.IMREAD_UNCHANGED)
            if img is not None:
                preloaded.append((p, img))

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
            contents=["mean","min","max","std","median"]
            Df=pd.DataFrame({cdir:pd.Series(all_results[cdir]).agg(contents) for cdir in all_results.keys()})
            print(Df)
            print(f"Total time: {time() - st:.2f} sec")
    if savepath != None:
        save_db(all_results,str(peadirpath),__file__.split("\\")[-1]+"_prototype",
                {"limb_wigth":limb_wigth,"secdiff":"diff(grad())","max_worker":max_workers,"max_chunk(b)":chunk_bytes}
                ,savepath,dyjest=dyjest)
    return all_results


if __name__ == "__main__":
    from tkinter.filedialog import askdirectory
    peadirpath = askdirectory(title="画像が入っているフォルダを選択してください")
    max_chunk_GB=5
    result = main(peadirpath, max_workers=None, limb_wigth=24, allp_num=1360, chunk_bytes=max_chunk_GB * 1024**3,debug=True)
    