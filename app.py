
import streamlit as st
from chatbot import filter_laptops

# Cấu hình trang
st.set_page_config(page_title="Laptop Chatbot", page_icon="💻")

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
user_input = st.chat_input("Nhập yêu cầu của bạn, ví dụ: Laptop Dell dưới 15 triệu, RAM 8GB...")

# Xử lý khi người dùng nhập và gửi
if user_input:
    # Thêm tin nhắn người dùng vào lịch sử
    st.session_state.chat_history.append({"role": "user", "message": user_input})

    # Hiển thị tin nhắn người dùng ngay lập tức
    with st.chat_message("user"):
        st.markdown(user_input)

    # Hiển thị trạng thái "Đang xử lý..."
    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("💬 Đang xử lý...")

    result = filter_laptops(user_input)

    # Xóa placeholder và thêm kết quả vào lịch sử
    placeholder.empty()
    st.session_state.chat_history.append({"role": "assistant", "message": result})

    # Hiển thị kết quả
    with st.chat_message("assistant"):
        st.markdown(result)

    # Rerun để cập nhật giao diện (nếu cần)
    st.rerun()