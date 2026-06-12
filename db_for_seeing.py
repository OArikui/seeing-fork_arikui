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

def save_db(result:dict,#解析結果{subdir1:[img1,img2...],subdir2:[...]...}
            peadirpath:str,#フォルダーパス_seeingのmainの引数,askdirnameの戻り値でよい
            analysis_program:str,#解析に使ったプログラム名,verなどを追記してもよい
            argument:dict=None,#解析のパラメータ、保存用
            db_path:str=None,#seeing_<projectname or programename>.db
            contents=["mean","min","max","std","median"],
            dyjest=False):
    # 1. 設定と事前準備
    if db_path is None:
        # 末尾に .db を補完
        db_path = (
            str(Path(__file__).resolve().parent.parent)
            + "\\seeing_"
            + datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
            + ".db"
        )

    peadir_name = peadirpath.split("\\")[-1]
    saved_at = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ==============================================================================
    # 【モード1】dyjest=True: 統計量のみを共通テーブルにスタック保存
    # ==============================================================================
    if dyjest:
        digest_table_name = "summary_digest"  # 共通のテーブル名
        subdirs = list(result.keys())

        # 各サブフォルダの統計量をまとめて計算
        df_stats = pd.DataFrame(
            {cdir: pd.Series(result[cdir]).agg(contents) for cdir in subdirs}
        )

        # ご指定のレイアウトに合わせて、行データ(リスト)を構築
        digest_rows = []

        # 1行目: プログラム名, 保存時間, 引数辞書(文字列化して1つのセルに集約)
        meta_row = [analysis_program, saved_at, str(argument) if argument else ""]
        digest_rows.append(meta_row)

        # 2行目: 1マス空けてからサブフォルダ名（横軸ヘッダー）
        header_row = [""] + subdirs
        digest_rows.append(header_row)

        # 3行目以降: 各統計量 (mean, min, max...)
        for metric in contents:
            metric_row = [metric] + df_stats.loc[metric].tolist()
            digest_rows.append(metric_row)

        # ブロックの区切りとして空行を1行追加
        digest_rows.append([])

        # 列数を最大値に揃えてDataFrame化
        max_cols = max(len(r) for r in digest_rows)
        padded_rows = [r + [None] * (max_cols - len(r)) for r in digest_rows]
        df_new_digest = pd.DataFrame(padded_rows)

        # SQLiteで不整合が起きないよう、一貫した列名(col_0, col_1...)を付与
        df_new_digest.columns = [f"col_{i}" for i in range(max_cols)]

        # DBへの書き込み処理
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # すでに共通テーブルが存在するか確認
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{digest_table_name}'"
        )
        if cursor.fetchone():
            # 存在する場合は、一度読み込んで今回のデータと結合(列数が違っても自動拡張される)
            df_old = pd.read_sql(f"SELECT * FROM {digest_table_name}", conn)
            df_total = pd.concat([df_old, df_new_digest], ignore_index=True)
        else:
            df_total = df_new_digest

        # テーブルを更新保存
        df_total.to_sql(
            digest_table_name, conn, if_exists="replace", index=False
        )
        conn.close()

        print(f"ダイジェストを追加保存しました: {digest_table_name}")
        return df_total, None

    # ==============================================================================
    # 【モード2】dyjest=False: 従来通りpeadirごとに全画像データを保存
    # ==============================================================================
    else:
        # プレビュー
        Df = pd.DataFrame(
            {
                cdir: pd.Series(result[cdir]).agg(contents)
                for cdir in result.keys()
            }
        )

        # 解析結果の DataFrame 作成（縦に画像名が並ぶ形）
        df_dict = {}
        for name, val in result.items():
            stats = pd.Series(val).agg(contents)
            img_names = [f"img{i:08d}" for i in range(1, len(val) + 1)]
            val_series = pd.Series(val, index=img_names)
            df_dict[name] = pd.concat([stats, val_series])

        df_results = pd.DataFrame(df_dict)

        # メタデータの DataFrame 作成
        meta_data = {"saved_at": saved_at, "analyser": analysis_program}
        if argument:
            meta_data.update(argument)
        df_info = pd.DataFrame([meta_data])

        # SQLite データベースへの保存
        results_table_name = f"{peadir_name}_results"
        info_table_name = f"{peadir_name}_info"

        conn = sqlite3.connect(db_path)

        # 【結果テーブル】保存
        df_results.to_sql(
            results_table_name,
            conn,
            if_exists="replace",
            index=True,
            index_label="id",
        )

        # 【メタデータテーブル】保存
        df_info.to_sql(info_table_name, conn, if_exists="replace", index=False)

        conn.close()

        print(f"保存完了: {results_table_name} と {info_table_name}")
        return df_results, df_info