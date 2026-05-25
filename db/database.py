# -*- coding: utf-8 -*-
"""
ひだまり健康チェック管理システム DB層（Ver3.7）
- SQLite接続
- WAL設定
- 読み書きトランザクション
- DataFrame保存/読込
- DB整合性チェック

画面・業務ロジックからDB処理を分離するためのモジュールです。
"""

from __future__ import annotations

import re
import sqlite3
import threading
from contextlib import contextmanager
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


DATA_DIR = Path("data")
HIDAMARI_DB_FILE = DATA_DIR / "hidamari_health.db"

DB_BUSY_TIMEOUT_MS = 10000
DB_WRITE_LOCK = threading.RLock()
DB_LAST_INTEGRITY_RESULT = {"checked_at": "", "ok": True, "messages": []}


def configure_database(data_dir: str | Path = "data", db_file: str | Path | None = None) -> None:
    """DB保存先を設定する。app.py側から起動時に呼び出してもよい。"""
    global DATA_DIR, HIDAMARI_DB_FILE
    DATA_DIR = Path(data_dir)
    HIDAMARI_DB_FILE = Path(db_file) if db_file is not None else DATA_DIR / "hidamari_health.db"
    ensure_dirs()


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(value: Any, default: str = "") -> str:
    try:
        if pd.isna(value):
            return default
    except Exception:
        pass
    text = str(value).strip()
    if text.lower() in ["nan", "none", "nat"]:
        return default
    return text


def validate_sqlite_identifier(name: str) -> str:
    """テーブル名などのSQLite識別子を安全に扱う。"""
    name = clean_text(name)
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise ValueError(f"SQLite識別子が不正です: {name}")
    return name


def apply_sqlite_pragmas(conn: sqlite3.Connection, for_write: bool = False) -> None:
    """SQLite安定化のための基本PRAGMAを全接続に適用する。"""
    conn.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS};")
    conn.execute("PRAGMA foreign_keys=ON;")
    conn.execute("PRAGMA temp_store=MEMORY;")
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    if for_write:
        conn.execute("PRAGMA wal_autocheckpoint=1000;")


def _connect_hidamari_sqlite(for_write: bool = False) -> sqlite3.Connection:
    """DB接続の唯一の入口。app.py側で直接 sqlite3.connect を呼ばない。"""
    ensure_dirs()
    conn = sqlite3.connect(
        HIDAMARI_DB_FILE,
        timeout=DB_BUSY_TIMEOUT_MS / 1000,
        check_same_thread=False,
        isolation_level=None,
    )
    apply_sqlite_pragmas(conn, for_write=for_write)
    return conn


@contextmanager
def hidamari_db_connection(for_write: bool = False):
    """読み取り・書き込み共通のDB接続コンテキスト。"""
    conn = _connect_hidamari_sqlite(for_write=for_write)
    try:
        yield conn
    finally:
        try:
            conn.close()
        except Exception:
            pass


@contextmanager
def hidamari_write_transaction():
    """保存処理専用。BEGIN IMMEDIATEで書き込み競合を早期に整理する。"""
    with DB_WRITE_LOCK:
        with hidamari_db_connection(for_write=True) as conn:
            try:
                conn.execute("BEGIN IMMEDIATE;")
                yield conn
                conn.commit()
                try:
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
                except Exception:
                    pass
            except Exception:
                try:
                    conn.rollback()
                except Exception:
                    pass
                raise


def get_hidamari_conn() -> sqlite3.Connection:
    """互換用の接続関数。新規コードではコンテキストマネージャを使う。"""
    return _connect_hidamari_sqlite(for_write=False)


def sqlite_table_exists(table_name: str) -> bool:
    table_name = validate_sqlite_identifier(table_name)
    ensure_dirs()
    if not HIDAMARI_DB_FILE.exists():
        return False
    try:
        with hidamari_db_connection(for_write=False) as conn:
            cur = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            return cur.fetchone() is not None
    except Exception:
        return False


def sqlite_table_row_count(table_name: str) -> int:
    table_name = validate_sqlite_identifier(table_name)
    if not sqlite_table_exists(table_name):
        return 0
    try:
        with hidamari_db_connection(for_write=False) as conn:
            cur = conn.execute(f'SELECT COUNT(*) FROM "{table_name}"')
            return int(cur.fetchone()[0])
    except Exception:
        return 0


def normalize_df_columns(df: pd.DataFrame | None, columns: list) -> pd.DataFrame:
    """DataFrameに必要列をそろえる。"""
    if df is None:
        df = pd.DataFrame(columns=columns)
    df = df.copy()
    for col in columns:
        if col not in df.columns:
            df[col] = ""
    return df[columns].astype("object")


def _to_sqlite_value(value: Any) -> Any:
    """SQLiteに保存しやすい値へ変換。"""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    if isinstance(value, (datetime, pd.Timestamp)):
        if pd.isna(value):
            return ""
        if value.hour == 0 and value.minute == 0 and value.second == 0:
            return value.strftime("%Y-%m-%d")
        return value.strftime("%Y-%m-%d %H:%M:%S")
    if isinstance(value, date):
        return value.strftime("%Y-%m-%d")
    if isinstance(value, bool):
        return int(value)
    return value


def prepare_sqlite_dataframe(
    df: pd.DataFrame,
    columns: list,
    date_cols: list | None = None,
    unique_cols: list | None = None,
    sort_cols: list | None = None,
) -> pd.DataFrame:
    """保存前のDataFrame整形を一本化する。"""
    date_cols = date_cols or []
    unique_cols = unique_cols or []
    sort_cols = sort_cols or []

    work = normalize_df_columns(df, columns)

    for col in date_cols:
        if col in work.columns:
            work[col] = pd.to_datetime(work[col], errors="coerce")

    if unique_cols and not work.empty:
        for col in unique_cols:
            if col in work.columns:
                work[col] = work[col].fillna("").astype(str).str.strip()
        work = work.drop_duplicates(subset=unique_cols, keep="last")

    if sort_cols and not work.empty:
        tmp_cols = []
        for col in sort_cols:
            if col in work.columns:
                tmp_col = col + "_sort_tmp"
                work[tmp_col] = pd.to_datetime(work[col], errors="coerce")
                tmp_cols.append(tmp_col)
        if tmp_cols:
            work = work.sort_values(tmp_cols, ascending=[False] * len(tmp_cols)).drop(columns=tmp_cols)

    work = work.astype("object")
    for col in work.columns:
        work[col] = work[col].map(_to_sqlite_value)
    return work


def db_write_dataframe(
    df: pd.DataFrame,
    table_name: str,
    columns: list,
    date_cols: list | None = None,
    unique_cols: list | None = None,
    sort_cols: list | None = None,
) -> None:
    """DB書き込みの一本化関数。全保存処理はここを通す。"""
    table_name = validate_sqlite_identifier(table_name)
    work = prepare_sqlite_dataframe(
        df,
        columns,
        date_cols=date_cols,
        unique_cols=unique_cols,
        sort_cols=sort_cols,
    )
    with hidamari_write_transaction() as conn:
        work.to_sql(table_name, conn, if_exists="replace", index=False)


def db_read_dataframe(table_name: str, columns: list, date_cols: list | None = None) -> pd.DataFrame:
    """DB読み込みの一本化関数。存在しない場合は空テーブルを作成する。"""
    table_name = validate_sqlite_identifier(table_name)
    date_cols = date_cols or []
    if not sqlite_table_exists(table_name):
        db_write_dataframe(pd.DataFrame(columns=columns), table_name, columns, date_cols=date_cols)

    try:
        with hidamari_db_connection(for_write=False) as conn:
            df = pd.read_sql_query(f'SELECT * FROM "{table_name}"', conn)
    except Exception:
        df = pd.DataFrame(columns=columns)

    df = normalize_df_columns(df, columns)
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df.astype("object")


def save_sqlite_table(
    df: pd.DataFrame,
    table_name: str,
    columns: list,
    date_cols: list | None = None,
    unique_cols: list | None = None,
    sort_cols: list | None = None,
) -> None:
    """互換用保存関数。内部では安定化済みDB書き込み関数を使う。"""
    db_write_dataframe(
        df,
        table_name,
        columns,
        date_cols=date_cols,
        unique_cols=unique_cols,
        sort_cols=sort_cols,
    )


def load_sqlite_table(table_name: str, columns: list, date_cols: list | None = None) -> pd.DataFrame:
    """互換用読込関数。内部では安定化済みDB読み込み関数を使う。"""
    return db_read_dataframe(table_name, columns, date_cols=date_cols)


def initialize_sqlite_engine() -> None:
    """起動時にWALモードを固定し、DBファイルを初期化する。"""
    ensure_dirs()
    with DB_WRITE_LOCK:
        with hidamari_db_connection(for_write=True) as conn:
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute(f"PRAGMA busy_timeout={DB_BUSY_TIMEOUT_MS};")
            conn.execute("PRAGMA wal_autocheckpoint=1000;")


def run_db_integrity_check(auto_repair: bool = True) -> dict:
    """SQLite自動整合性チェック。quick_checkがNGの場合は警告を返す。"""
    global DB_LAST_INTEGRITY_RESULT
    messages: list[str] = []
    ok = True
    checked_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        initialize_sqlite_engine()
        with hidamari_db_connection(for_write=False) as conn:
            quick = conn.execute("PRAGMA quick_check;").fetchone()
            quick_result = quick[0] if quick else "unknown"
            if str(quick_result).lower() != "ok":
                ok = False
                messages.append(f"quick_check: {quick_result}")
            else:
                messages.append("quick_check: ok")

            journal = conn.execute("PRAGMA journal_mode;").fetchone()
            messages.append(f"journal_mode: {journal[0] if journal else 'unknown'}")

        if auto_repair and ok:
            with hidamari_write_transaction() as conn:
                try:
                    conn.execute("ANALYZE;")
                except Exception:
                    pass
                try:
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE);")
                except Exception:
                    pass
    except Exception as e:
        ok = False
        messages.append(f"整合性チェックエラー: {e}")

    DB_LAST_INTEGRITY_RESULT = {"checked_at": checked_at, "ok": ok, "messages": messages}
    return DB_LAST_INTEGRITY_RESULT


def get_last_integrity_result() -> dict:
    return DB_LAST_INTEGRITY_RESULT


def get_db_integrity_status_text() -> str:
    result = DB_LAST_INTEGRITY_RESULT or {}
    status = "OK" if result.get("ok", True) else "要確認"
    checked_at = result.get("checked_at", "未実施")
    messages = " / ".join(result.get("messages", []))
    return f"DB整合性: {status}（{checked_at}） {messages}"
