import sqlite3
from datetime import datetime
import uuid

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
            created_at TEXT
        )
    """)
    conn.commit()
    conn.close()

def save_chat_session(session_id, chat_history):
    """
    Save the entire chat history for a session_id, replacing any existing records for that session_id.
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()

    # Xóa các bản ghi cũ của session_id để tránh trùng lặp
    c.execute("DELETE FROM chat_sessions WHERE session_id = ?", (session_id,))

    # Lưu từng cặp hỏi-đáp
    created_at = datetime.now().isoformat()
    for i in range(0, len(chat_history), 2):  # Assumes user-assistant pairs
        if i + 1 < len(chat_history):  # Ensure there's a pair
            user_msg = chat_history[i]["message"] if chat_history[i]["role"] == "user" else None
            assistant_msg = chat_history[i+1]["message"] if chat_history[i+1]["role"] == "assistant" else None
            if user_msg and assistant_msg:
                c.execute("""
                    INSERT INTO chat_sessions (session_id, question, answer, created_at)
                    VALUES (?, ?, ?, ?)
                """, (session_id, user_msg, assistant_msg, created_at))
    
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
    c.execute("SELECT question, answer FROM chat_sessions WHERE session_id = ? ORDER BY id", (session_id,))
    history = c.fetchall()
    conn.close()
    
    # Chuyển đổi thành định dạng chat_history
    chat_history = []
    for question, answer in history:
        chat_history.append({"role": "user", "message": question})
        chat_history.append({"role": "assistant", "message": answer})
    return chat_history