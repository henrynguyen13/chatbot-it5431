import re
import uuid
from datetime import datetime

import streamlit as st

from chatbot import process_query, clear_memory, load_memory_from_history
from history import init_db, save_chat_session, get_chat_sessions, load_chat_session
from intents import handle_buy_laptop
from invoice import init_invoice_table

# Khởi tạo DB
init_db()
init_invoice_table()

# Cấu hình trang
st.set_page_config(page_title="Laptop Chatbot", page_icon="💻")

# print("session_state:", st.session_state)

# Thêm nút ẩn để kích hoạt form
st.markdown("""
<button id="buy-laptop-btn" style="display:none;"></button>
<input type="hidden" id="laptop-name">
""", unsafe_allow_html=True)

# Thêm form đặt hàng (ẩn ban đầu)
if "show_buy_form" not in st.session_state:
    st.session_state.show_buy_form = False

if st.session_state.show_buy_form:
    laptop_name = st.session_state.buy_laptop_name
    handle_buy_laptop(laptop_name)

# Khởi tạo lịch sử chat và session_id nếu chưa có
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "filter_history" not in st.session_state:
    st.session_state.filter_history = []

# Sidebar
st.sidebar.title("🛠 Chat History")

# Nút tạo phiên chat mới
if st.sidebar.button("🆕 New Chat"):
    st.session_state.show_buy_form = False
    st.session_state.chat_history = []
    st.session_state.session_id = str(uuid.uuid4())  # Tạo session_id mới
    st.session_state.filter_history = []
    clear_memory()
    st.rerun()

# Hiển thị danh sách phiên đã lưu
sessions = get_chat_sessions()
# print("sessions:", sessions)
if sessions:
    st.sidebar.markdown("### Recent Sessions")
    # Sắp xếp các phiên theo created_at giảm dần (gần nhất trước)
    for session_id, created_at in sorted(sessions, key=lambda x: x[1], reverse=True):
        # Định dạng thời gian cho dễ đọc
        try:
            formatted_time = datetime.fromisoformat(created_at).strftime("%b %d, %Y %H:%M")
        except ValueError:
            formatted_time = created_at
        
        # Tạo nút cho mỗi phiên
        session_label = f"Session {session_id[:8]} - {formatted_time}"
        if st.sidebar.button(session_label, key=f"session_{session_id}"):
            if session_id != st.session_state.session_id:
                st.session_state.chat_history, st.session_state.filter_history = load_chat_session(session_id)
                st.session_state.session_id = session_id
                load_memory_from_history(st.session_state.chat_history)
                st.rerun()
else:
    st.sidebar.text("No previous sessions found.")

# Tiêu đề
st.title("💻 Chatbot Laptop Advisor")

# Hiển thị lịch sử chat
for chat in st.session_state.chat_history:
    with st.chat_message(chat["role"]):
        st.markdown(chat["message"])

# Ô nhập liệu chat
user_input = st.chat_input("Enter your message, example: Laptop Dell under 15 million VND, RAM 8GB...")

# Xử lý khi người dùng nhập và gửi
if user_input:
    # Thêm tin nhắn người dùng vào lịch sử
    st.session_state.chat_history.append({"role": "user", "message": user_input})

    # Hiển thị tin nhắn người dùng
    with st.chat_message("user"):
        st.markdown(user_input)

    # Xử lý truy vấn
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("💬 Processing...")
        result = process_query(user_input, st.session_state.filter_history)

        # Kiểm tra nếu kết quả chứa mã kích hoạt form
        if "<script>" in result:
            st.markdown(result, unsafe_allow_html=True)
            st.session_state.show_buy_form = True
            laptop_name = re.search(r"document.getElementById\('laptop-name'\).value = '(.*?)'", result).group(1)
            st.session_state.buy_laptop_name = laptop_name
        else:
            placeholder.empty()
            st.markdown(result)

    # Thêm câu trả lời vào lịch sử
    st.session_state.chat_history.append({"role": "assistant", "message": result})

    # Lưu tự động vào cơ sở dữ liệu
    save_chat_session(st.session_state.session_id, st.session_state.chat_history, st.session_state.filter_history)
    st.success("✅ Session saved automatically.")

    # Cập nhật giao diện
    st.rerun()