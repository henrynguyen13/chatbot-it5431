import sqlite3
from datetime import datetime
import json

DB_NAME = "chat_history.db"

def init_db():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            question TEXT,
            answer TEXT,
            filters TEXT,
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_chat_session(session_id, chat_history, filter_history):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    created_at = datetime.now().isoformat()
    for i in range(len(chat_history) - 2, len(chat_history), 2):  # Chỉ lưu các mục mới
        if i + 1 < len(chat_history):
            user_msg = chat_history[i]["message"] if chat_history[i]["role"] == "user" else None
            assistant_msg = chat_history[i+1]["message"] if chat_history[i+1]["role"] == "assistant" else None

            filter_idx = i // 2  # Mỗi cặp user-assistant tương ứng với một bộ lọc
            filters_json = json.dumps(filter_history[filter_idx]) if filter_idx < len(filter_history) else "{}"
            if user_msg and assistant_msg:
                c.execute("""
                    INSERT INTO chat_sessions (session_id, question, answer, filters, created_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (session_id, user_msg, assistant_msg, filters_json, created_at))
    conn.commit()
    conn.close()

def get_chat_sessions():
    """Lấy danh sách tất cả session_id từ cơ sở dữ liệu."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT DISTINCT session_id, MAX(created_at) FROM chat_sessions GROUP BY session_id ORDER BY created_at DESC")
    sessions = c.fetchall()
    conn.close()
    return sessions

def load_chat_session(session_id):
    """Tải lịch sử hội thoại của một phiên từ cơ sở dữ liệu."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute("SELECT question, answer, filters FROM chat_sessions WHERE session_id = ? ORDER BY id", (session_id,))
    history = c.fetchall()
    conn.close()
    
    # Chuyển đổi thành định dạng chat_history
    chat_history = []
    filter_history = []
    for question, answer, filters_json in history:
        chat_history.append({"role": "user", "message": question})
        chat_history.append({"role": "assistant", "message": answer})
        filter_history.append(json.loads(filters_json) if filters_json else {})
    return chat_history, filter_history