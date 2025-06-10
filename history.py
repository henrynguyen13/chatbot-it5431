# db.py
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
    Save each user-assistant pair to DB with same session_id
    """
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    created_at = datetime.now().isoformat()

    for i in range(0, len(chat_history) - 1, 2):  # Assumes user-assistant pairs
        user_msg = chat_history[i]["message"] if chat_history[i]["role"] == "user" else None
        assistant_msg = chat_history[i+1]["message"] if chat_history[i+1]["role"] == "assistant" else None
        if user_msg and assistant_msg:
            c.execute("""
                INSERT INTO chat_sessions (session_id, question, answer, created_at)
                VALUES (?, ?, ?, ?)
            """, (session_id, user_msg, assistant_msg, created_at))
    conn.commit()
    conn.close()
