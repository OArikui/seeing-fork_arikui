import os
import numpy as np
import pandas as pd
import sqlite3
import datetime
from pathlib import Path
def who_called_me():
    import inspect
    caller = inspect.stack()[1]          # 1 は「呼び出し元」
    filename = caller.filename           # フルパス
    return os.path.basename(filename)    # ファイル名だけ返す

def largest_subdir_files(root):
    max_count = -1
    winner = None

    for name in os.listdir(root):
        sub = os.path.join(root, name)
        if os.path.isdir(sub):
            entries = os.listdir(sub)
            count = sum(1 for e in entries if os.path.isfile(os.path.join(sub, e)))
            if count > max_count:
                max_count = count
                winner = sub
    if winner is None:
        return 0, []
    files = [f for f in os.listdir(winner)
             if os.path.isfile(os.path.join(winner, f))]

    return max_count, files

def save_db(result,peadirpath,argument={},savepath=None,contents=["metadata","mean","min","max","std","median"],extention=[".tiff"]):
    rows = []            
    Df=pd.DataFrame({cdir:pd.Series(result[cdir]).agg(contents) for cdir in result.keys()})
    print(rows)
    #保存時はこの下にlarget_subdir_filesで全データを書き込む
    # DataFrame 作成（元コードに合わせた出力）
    p = Path(peadirpath + "\\" + os.listdir(peadirpath)[0])  # 最初のサブディレクトリからファイル名を取得、subdirのfile数が違うならlargest_subdir_filesを使って
    df_dict = {"content": contents +[f.name for f in p.iterdir() if f.is_file() and f.suffix.lower() in extention]}
    arg=""
    if len(argument.items()) > 0:
        for key,val in argument.items():
            arg+=str(key)+":"+str(val)+","
        arg=arg[:-1]
    meta=who_called_me()+"_"+arg+"_"+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    for i,(name, vals) in enumerate(result.items()):
        if i>0:
            meta=""
        if len(vals) == 0:
            df_dict[name] = [meta]+[np.nan] * len(df_dict["content"])
        else:
            arr = np.array(vals, dtype=np.float64)
            df_dict[name] = [meta]+pd.Series(result[name]).agg(contents[1:])+ arr.tolist()
    Df = pd.DataFrame(df_dict)
    if savepath:
        tabel=peadirpath.split("\\")[-1]
        with sqlite3.connect(savepath) as conn:
            Df.to_sql(tabel, conn, if_exists="append", index=False, method="multi", chunksize=500)
        print("saved db",savepath)

