import streamlit as st
from chatbot import process_query, clear_memory, load_memory_from_history
import uuid
from history import init_db, save_chat_session, get_chat_sessions, load_chat_session

# Khởi tạo DB
init_db()

# Cấu hình trang
st.set_page_config(page_title="Laptop Chatbot", page_icon="💻")

# Khởi tạo lịch sử chat và session_id nếu chưa có
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

# Sidebar
st.sidebar.title("🛠 Chat History")

# Nút tạo phiên chat mới
if st.sidebar.button("🆕 New Chat"):
    st.session_state.chat_history = []
    st.session_state.session_id = str(uuid.uuid4())  # Tạo session_id mới
    clear_memory()
    st.rerun()

# Hiển thị danh sách phiên đã lưu
sessions = get_chat_sessions()
if sessions:
    session_options = [f"Session {i+1} ({s[1]})" for i, s in enumerate(sessions)]
    selected_session = st.sidebar.selectbox(
        "📜 Load previous session",
        ["None"] + session_options,
        key="session_select"  # Đảm bảo selectbox có key duy nhất
    )
    
    if selected_session != "None":
        selected_session_id = sessions[session_options.index(selected_session) - 1][0]
        if selected_session_id != st.session_state.session_id:  # Chỉ tải nếu session_id khác
            st.session_state.chat_history = load_chat_session(selected_session_id)
            st.session_state.session_id = selected_session_id
            load_memory_from_history(st.session_state.chat_history)
            st.rerun()
else:
    st.sidebar.text("No previous sessions found.")

# Tiêu đề
st.title("💻 Chatbot Tư vấn Mua Laptop")

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
        result = process_query(user_input)
        placeholder.empty()
        st.markdown(result)

    # Thêm câu trả lời vào lịch sử
    st.session_state.chat_history.append({"role": "assistant", "message": result})

    # Lưu tự động vào cơ sở dữ liệu
    save_chat_session(st.session_state.session_id, st.session_state.chat_history)
    st.success("✅ Session saved automatically.")

    # Cập nhật giao diện
    st.rerun()