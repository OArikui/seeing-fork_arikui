#!/usr/bin/env python3
# bench_inline.py
import sqlite3
import time
import json
import traceback
from datetime import datetime
from typing import Any, Callable, List, Sequence

DB_PATH = "results.db"          # 保存先DBファイル（必要に応じて変更）
REPEATS = 5                     # 各組み合わせの繰り返し回数
WARMUP = 1                      # 各組み合わせのウォームアップ回数
MIN_TIME = 0.0                  # 0.0なら無効。>0なら合計実行時間がこの秒数以上になるまで繰り返す

"""
使い方
・比較したい関数が別モジュールにある場合は上部で from myfuncs import f1, f2 のように import して FUNCTIONS に追加する。
・ARG_PATTERNS に試したい引数パターンをリストで書く（位置引数のみ）。
・必要なら DB_PATH / REPEATS / WARMUP / MIN_TIME を編集する。

dbの形式

カラム名	       型 	     説明    	                        例
id	              INTEGER	自動採番の主キー	                 1
timestamp	      TEXT	    実行時刻（UTC ISO）	                2026-06-06T09:12:34.123456Z
module	          TEXT	    関数が属するモジュール名（空文字可）  myfuncs
function	      TEXT	    関数名	                            f1
args_json	      TEXT	    実行時の引数を JSON 文字列で保存     "[1, 2]"
repeat_index      INTEGER	繰り返しインデックス（0 から）	       0
duration_seconds  REAL	    実行時間（秒）。失敗時は NULL	      0.000345
success	          INTEGER	成功なら 1、失敗なら 0	                1
error_text	      TEXT	    例外のトレース（失敗時のみ）    	"Traceback: ..."
"""

# --- ここにベンチ対象の関数を import してリストに入れる ---
# 例: from myfuncs import f1, f2, f3
# FUNCTIONS = [f1, f2, f3]

# 代わりにスクリプト内で関数を定義してもOK
def example_fast(a, b):
    s = 0
    for i in range(10000):
        s += (a + b + i) % 7
    return s

def example_slow(a, b):
    time.sleep(0.001)
    return example_fast(a, b)

# ベンチする関数のリスト（callable を直接入れる）
FUNCTIONS: List[Callable[..., Any]] = [
    example_fast,
    example_slow,
]

# --- ここに試す引数パターンを定義する ---
# 各要素は位置引数のリスト。キーワード引数が必要なら下の拡張例を参照
ARG_PATTERNS: List[Sequence[Any]] = [
    [1, 2],
    [10, 20],
    [100, 200],
]

# --- DB スキーマ ---
DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    module TEXT,
    function TEXT NOT NULL,
    args_json TEXT NOT NULL,
    repeat_index INTEGER NOT NULL,
    duration_seconds REAL,
    success INTEGER NOT NULL,
    error_text TEXT
);
CREATE INDEX IF NOT EXISTS idx_fn ON runs(function);
"""

def ensure_db(path: str):
    conn = sqlite3.connect(path)
    cur = conn.cursor()
    cur.executescript(DB_SCHEMA)
    conn.commit()
    conn.close()

def record_run(db_path: str, module: str | None, func_name: str, args: Any, repeat_index: int,
               duration: float | None, success: bool, error_text: str | None):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO runs (timestamp, module, function, args_json, repeat_index, duration_seconds, success, error_text) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (datetime.utcnow().isoformat() + "Z",
         module if module is not None else "",
         func_name,
         json.dumps(args, ensure_ascii=False),
         repeat_index,
         duration if duration is not None else None,
         1 if success else 0,
         error_text)
    )
    conn.commit()
    conn.close()

def call_and_time(func: Callable[..., Any], args: Sequence[Any]) -> tuple[float, Any]:
    start = time.perf_counter()
    result = func(*args)
    end = time.perf_counter()
    return end - start, result

def success_check(return_value):#成功か否かを判定できます
    if return_value:
        return True
    else:
        return False
def main():
    ensure_db(DB_PATH)
    for func in FUNCTIONS:
        func_name = getattr(func, "__name__", repr(func))
        module_name = getattr(func, "__module__", None)
        for pattern in ARG_PATTERNS:
            # Warmup
            for _ in range(WARMUP):
                try:
                    func(*pattern)
                except Exception:
                    pass

            for repeat_index in range(REPEATS):
                total_time = 0.0
                runs = 0
                while True:
                    try:
                        duration, result = call_and_time(func, pattern)
                        success = success_check(result)
                        error_text = None
                    except Exception:
                        duration = None
                        success = False
                        error_text = traceback.format_exc()
                    record_run(DB_PATH, module_name, func_name, pattern, repeat_index, duration, success, error_text)
                    runs += 1
                    if duration is not None:
                        total_time += duration
                    if MIN_TIME <= 0.0:
                        break
                    if total_time >= MIN_TIME:
                        break
                print(f"[{func_name}] args={pattern} repeat={repeat_index} runs={runs} total_time={total_time:.6f}s")

if __name__ == "__main__":
    main()
