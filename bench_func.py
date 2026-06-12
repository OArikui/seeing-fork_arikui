#!/usr/bin/env python3
# bench_inline.py
import sqlite3
import time
import json
import traceback
from datetime import datetime, timedelta
from typing import Any, Callable, List, Sequence

db_title = "seeing_bench"
DB_PATH = __file__.replace(__file__.split("\\")[-1], "data_souces\\" + db_title) + ".db"  # 保存先DBファイル（必要に応じて変更）
REPEATS = 2                     # 各組み合わせの繰り返し回数
WARMUP = 0                      # 各組み合わせのウォームアップ回数
MIN_TIME = 0.0                  # 0.0なら無効。>0なら合計実行時間がこの秒数以上になるまで繰り返す

# --- ここにベンチ対象の関数を import してリストに入れる --
from see_testers.fin_seeing_allframe import main as fin_seeing_allframe
from see_testers.fin_seeingonefram_propool import main as fin_seeingonefram_propool
from see_testers.fin_seeingonefram_propool_chunk import main as fin_seeingonefram_propool_chunk
from see_testers.fin_seeingonefram_propool_chunk_fast import main as fin_seeingonefram_propool_chunk_fast

# ベンチする関数のリスト（callable を直接入れる）
FUNCTIONS: List[Callable[..., Any]] = [
    fin_seeing_allframe,
    fin_seeingonefram_propool,
    fin_seeingonefram_propool_chunk,
    fin_seeingonefram_propool_chunk_fast
]

# --- ここに試す引数パターンを定義する ---
test_save = r"C:\projects\seeing-fork_arikui\see_testers\test_result.db"
ARG_PATTERNS: List[Sequence[Any]] = [
    [r"J:\2025-08-30Z\2025-08-30-LT", test_save, True],
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

def success_check(return_value):
    if return_value:
        return True
    else:
        return False

def main():
    ensure_db(DB_PATH)
    
    total_steps = len(FUNCTIONS) * len(ARG_PATTERNS) * REPEATS
    current_step = 0
    start_time = time.perf_counter()
    
    print(f"ベンチマーク開始: 全 {total_steps} ステップ (Warmup除く)")
    
    for func in FUNCTIONS:
        func_name = getattr(func, "__name__", repr(func))
        module_name = getattr(func, "__module__", None)
        
        short_module = module_name.split('.')[-1] if module_name else ""
        display_name = f"{short_module}.{func_name}" if short_module else func_name
        
        # モジュール実行の区切りを表示
        print(f"\n========== 【モジュール】 {display_name} ==========")
        
        for pattern in ARG_PATTERNS:
            # 現在テストしている変数を表示
            print(f"  [テスト変数] args: {pattern}")
            
            # Warmup
            if WARMUP > 0:
                for w in range(WARMUP):
                    # ウォームアップの状況を同一行で上書き表示（見栄え用）
                    print(f"  -> ウォームアップ ({w + 1}/{WARMUP}) 実行中...", end="\r")
                    try:
                        func(*pattern)
                    except Exception:
                        pass
                print(f"  -> ウォームアップ 完了{' ' * 15}") # 上書き消去用スペース付き

            print("  -> 本番計測開始...")
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
                
                # 進捗計算
                current_step += 1
                elapsed = time.perf_counter() - start_time
                
                avg_time_per_step = elapsed / current_step
                remaining_steps = total_steps - current_step
                eta_seconds = avg_time_per_step * remaining_steps
                
                eta_td = timedelta(seconds=int(eta_seconds))
                estimated_end = datetime.now() + eta_td
                
                progress_pct = (current_step / total_steps) * 100
                
                # モジュール名や変数は上で表示しているので、ここでは計測結果と進捗のみをスッキリと表示
                print(
                    f"    [{current_step}/{total_steps}] ({progress_pct:5.1f}%) "
                    f"repeat={repeat_index + 1}/{REPEATS} total_time={total_time:.6f}s "
                    f"| 残り約: {eta_td} (終了予想: {estimated_end.strftime('%H:%M:%S')})"
                )

if __name__ == "__main__":
    main()