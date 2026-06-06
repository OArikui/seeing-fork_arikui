import os
from pathlib import Path
import cv2
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from decimal import Decimal, ROUND_HALF_UP
from MIN2_ignore_sunspots import MIN2_ignore_sunspots as MIN2_ver1

def seeing_one_frame_fast(readed_img, cir_stat, limb_wigth=24, allp_num=1360):
    """
    高速化版 seeing_one_frame
    - Decimal を使わず numpy の丸めを使用
    - allp_num は 4 の倍数であることを前提
    """
    (cx,cy),r = cir_stat
    if allp_num % 4 != 0:
        raise ValueError("allp_numは4の倍数にしてください")

    # i の配列（整数）
    half = int(allp_num / 4 / 2)
    i_vals = np.arange(-half, half, dtype=np.float64)  # float で計算

    # min2r を配列で計算（近似円）
    # sqrt(r^2 - i^2) が負になる可能性を clamp
    tmp = r * r - i_vals * i_vals
    tmp[tmp < 0] = 0.0
    min2r_arr = np.sqrt(tmp)
    min2rlst = []

    realrlst = []

    for idx, min2r in enumerate(min2r_arr):
        min2r_int = float(Decimal(min2r).quantize(Decimal("1"), rounding=ROUND_HALF_UP))
        min2rlst.append(float(Decimal(min2r).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)))

        # L
        y = int((cy + i_vals[idx]))
        x_start = int((cx - min2r_int - limb_wigth))
        x_end = int((cx - min2r_int + limb_wigth))
        samples = readed_img[y, x_start:x_end]
        if samples.size == 0:
            raise ("sample size is zero")
        else:
            grad = np.gradient(samples.astype(np.float64))#一回微分
            diff = np.diff(grad)#２回微分
            # argmax が空配列になる可能性を考慮
            if diff.size == 0:
                raise ("sample size is zero")
            else:
                realindx = float(np.argmax(diff)) + 0.5#diffで減る分の0.5
        realrlst.append(min2r_int + realindx - limb_wigth)

        # R
        x_start = int((cx + min2r_int - limb_wigth))
        x_end = int((cx + min2r_int + limb_wigth))
        samples = readed_img[y, x_start:x_end]
        if samples.size == 0:
            raise ("sample size is zero")
        else:
            grad = np.gradient(samples.astype(np.float64))
            diff = np.diff(grad)
            if diff.size == 0:
                raise ("sample size is zero")
            else:
                realindx = float(np.argmax(diff)) + 0.5#diffで減る分の0.5
        realrlst.append(min2r_int + realindx - limb_wigth)

        # T
        x = int((cx + i_vals[idx]))
        y_start = int((cy - min2r_int - limb_wigth))
        y_end = int((cy - min2r_int + limb_wigth))
        samples = readed_img[y_start:y_end, x]
        if samples.size == 0:
            raise ("sample size is zero")
        else:
            grad = np.gradient(samples.astype(np.float64))
            diff = np.diff(grad)
            if diff.size == 0:
                raise ("sample size is zero")
            else:
                realindx = float(np.argmax(diff)) + 0.5#diffで減る分の0.5
        realrlst.append(min2r_int + realindx - limb_wigth)

        # B
        y_start = int((cy + min2r_int - limb_wigth))
        y_end = int((cy + min2r_int + limb_wigth))
        samples = readed_img[y_start:y_end, x]
        if samples.size == 0:
            raise ("sample size is zero")
        else:
            grad = np.gradient(samples.astype(np.float64))
            diff = np.diff(grad)
            if diff.size == 0:
                raise ("sample size is zero")
            else:
                realindx = float(np.argmax(diff)) + 0.5#diffで減る分の0.5
        realrlst.append(min2r_int + realindx - limb_wigth)

    realrlst = np.array(realrlst, dtype=np.float64)
    min2rlst_rep = np.repeat(min2rlst, 4)
    return float(np.std(realrlst - min2rlst_rep))


def process_single_image(path, limb_wigth=24, allp_num=1360):
    """
    1画像分の処理を行い、seeing 値を返す。
    path は画像ファイルのフルパス
    """
    # 画像読み込み
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if img is None:
        return None  # 読み込めなかった

    # MIN2_ver1 はグレースケールを期待しているので合わせる
    # ((readed_img >> 8).astype("uint8")) 
    try:
        gray_for_min2 = ((img >> 8).astype("uint8"))
    except Exception:
        # もし既に 8bit ならそのまま
        gray_for_min2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img

    # cir_stat を取得
    try:
        cir_stat = MIN2_ver1(gray_for_min2, n=10, light_threshold=50, limb_wigth=limb_wigth)
    except Exception:
        return None

    # seeing を計算
    try:
        seeing_val = seeing_one_frame_fast(img, cir_stat, limb_wigth=limb_wigth, allp_num=allp_num)
    except Exception:
        return None

    return seeing_val


def gather_image_paths(root_dir, exts=(".jpg", ".jpeg", ".png", ".tiff")):
    root = Path(root_dir)
    groups = {}
    for sub in root.iterdir():
        if sub.is_dir():
            files = [p for p in sub.iterdir() if p.suffix.lower() in exts]
            if files:
                groups[sub.name] = files
    return groups


def main(peadirpath, max_workers=None, limb_wigth=24, allp_num=1360,debug=False):
    print(f"--seeing_main_Debug--\nselected:{peadirpath}\nmax_workers:{max_workers}")
    groups = gather_image_paths(peadirpath)
    results = {name: [] for name in groups}
    # 全ファイル数を数えて tqdm の total に使う
    total_files = sum(len(v) for v in groups.values())

    # 並列実行（プロセスプール）
    with ProcessPoolExecutor(max_workers=max_workers) as exe:
        # submit して future を管理する方法で進捗を表示
        futures = {}
        for group_name, files in groups.items():
            for p in files:
                fut = exe.submit(process_single_image, str(p), limb_wigth, allp_num)
                futures[fut] = group_name

        for fut in tqdm(as_completed(futures), total=total_files, desc="Processing images"):
            group_name = futures[fut]
            try:
                val = fut.result()
            except Exception:
                val = None
            if val is not None:
                results[group_name].append(val)
    return results


if __name__ == "__main__":
    from tkinter.filedialog import askdirectory
    from time import time
    import pandas as pd
    peadirpath = askdirectory(title="画像が入っているフォルダを選択してください")
    st = time()
    # max_workers を None にすると CPU コア数に合わせる
    results = main(peadirpath, max_workers=None, limb_wigth=24, allp_num=1360,debug=True)#resulltはdict
    contents=["mean","min","max","std","median"]
    Df=pd.DataFrame({cdir:pd.Series(results[cdir]).agg(contents) for cdir in results.keys()})
    print(Df)
    print(f"seeing_one_frame の解析時間: {time() - st:.2f} 秒")
    