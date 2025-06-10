
import streamlit as st

from chatbot import process_query
import uuid

from history import init_db, save_chat_session

# Khởi tạo DB
init_db()

# Cấu hình trang
st.set_page_config(page_title="Laptop Chatbot", page_icon="💻")

# Sidebar
if st.sidebar.button("🆕 New Chat"):
    st.session_state.chat_history = []
    st.session_state.session_id = str(uuid.uuid4())
    st.rerun()

st.sidebar.title("🛠 Chat History")

# Nút lưu
if st.sidebar.button("💾 Save session chat"):
    if st.session_state.chat_history:
        save_chat_session(st.session_state.session_id, st.session_state.chat_history)
        st.success("✅ Đã lưu phiên chat vào cơ sở dữ liệu")
    else:
        st.warning("⚠️ Không có nội dung để lưu")

# Tiêu đề
st.title("💻 Chatbot Tư vấn Mua Laptop")

# Khởi tạo lịch sử chat nếu chưa có
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

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

    # Hiển thị tin nhắn người dùng ngay lập tức
    with st.chat_message("user"):
        st.markdown(user_input)

    # Hiển thị trạng thái "Processing..."
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("💬 Processing...")

    result = process_query(user_input)

    # Xóa placeholder và thêm kết quả vào lịch sử
    placeholder.empty()
    st.session_state.chat_history.append({"role": "assistant", "message": result})

    # Hiển thị kết quả
    with st.chat_message("assistant"):
        st.markdown(result)

    # Rerun để cập nhật giao diện (nếu cần)
    st.rerun()