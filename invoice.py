import sqlite3
from history import DB_NAME  # Dùng lại DB_NAME = "chat_history.db"

def init_invoice_table():
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS invoice (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            customer TEXT NOT NULL,
            laptop TEXT NOT NULL,
            phone TEXT NOT NULL,
            address TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def insert_invoice(customer, laptop, phone, address):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    c.execute('''
        INSERT INTO invoice (customer, laptop, phone, address)
        VALUES (?, ?, ?, ?)
    ''', (customer, laptop, phone, address))
    conn.commit()
    conn.close()
