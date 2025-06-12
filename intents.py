import re
import pandas as pd

def detect_intent(query: str) -> str:
    """Phát hiện intent từ câu hỏi người dùng với độ chính xác cao hơn"""
    query = query.lower().strip()

    # Intent chào hỏi
    if re.match(
            r"^(hi|hello|hey|greetings|good morning|good afternoon|good evening|what's up|sup|yo|hi there|hi chatbot|hi laptop bot)",
            query):
        return "greeting"

    # Intent cảm ơn
    elif re.match(r"^(thanks|thank you|thank you so much|appreciate it|cheers|nice one)", query):
        return "thanks"

    # Intent tạm biệt
    elif re.match(r"^(bye|goodbye|see you|farewell|cya|exit|quit|close|stop|end)", query):
        return "goodbye"

    # Intent giới thiệu bản thân
    elif re.match(r"^(who are you|what are you|introduce yourself|tell me about yourself|your name|what is your name)",
                  query):
        return "self_introduction"

    # Intent hỏi chức năng
    elif re.match(
            r"^(what can you do|how can you help|your capabilities|what do you do|help me|assist me|support|your purpose)",
            query):
        return "capabilities"

    elif re.search(r'\b(compare|vs|versus|difference between)\b', query):
        return "compare"

    elif re.search(r'\b(pick one|which one|recommend one|choose between|should i choose)\b', query):
        return "pick_one"


    elif re.search(
            r"(will choose|will buy|confirm to buy|order|confirm to order|want to order|i want to order|i want to buy)\s+(asus|hp|lenovo|dell|msi|acer|apple|macbook|vivobook|ideapad|inspiron|swift|xps|thinkpad|surface)[\w\s-]*",
            query, re.IGNORECASE
    ):
        return "buy_laptop"

    # Intent tìm kiếm - phát hiện các từ khóa liên quan đến laptop và thông số kỹ thuật
    laptop_keywords = [
        "laptop", "notebook", "computer", "pc", "device",
        "recommend", "suggest", "find", "search", "looking for",
        "buy", "purchase", "need", "want", "choose", "select",
        "spec", "specs", "specification", "configuration",
        "ram", "memory", "storage", "ssd", "hdd", "gb",
        "processor", "cpu", "gpu", "graphics", "screen", "display",
        "price", "cost", "budget", "usd", "dollar", "under", "below", "over", "above", "vnd"
        "brand", "model", "dell", "hp", "lenovo", "asus", "apple", "acer", "msi"
    ]

    # Phát hiện số kèm theo đơn vị (8GB, 500GB, $1000, etc.)
    has_spec_numbers = re.search(r'\b(\d+\s*(gb|gb ram|ram|ssd|hdd|tb|usd|\$))\b', query)

    # Phát hiện từ khóa liên quan đến laptop
    has_laptop_keyword = any(keyword in query for keyword in laptop_keywords)

    # Phát hiện cấu trúc câu hỏi tìm kiếm điển hình
    has_search_pattern = re.search(r'(laptop|notebook).*(for|with|under|around|that|over)', query)

    if has_spec_numbers or has_laptop_keyword or has_search_pattern:
        return "search"

    # Không xác định được intent
    return "unknown"


def handle_intent(intent: str, user_request: str = "") -> str:
    responses = {
        "greeting": "Hello! I'm your laptop shopping assistant. 🤖 How can I help you find the perfect laptop today?",
        "thanks": "You're welcome! 😊 If you need more help finding a laptop, just let me know!",
        "goodbye": "Goodbye! 👋 Feel free to come back if you need laptop recommendations anytime!",
        "self_introduction": "I'm a specialized AI assistant for laptop recommendations. My job is to help you find the perfect laptop based on your needs, budget, and preferences!",
        "capabilities": "I can help you:\n- Find laptops matching your budget 💰\n- Recommend laptops for specific uses (gaming, work, etc.) 🎮💼\n- Compare laptop specifications 📊\n- Suggest the best value laptops ⭐\n\nJust tell me what you're looking for!",
        "unknown": "I'm here to help you find the perfect laptop! 🚀 Could you tell me what kind of laptop you're looking for?"
    }

    return responses.get(intent, responses["unknown"])

def handle_compare(user_request: str, df: pd.DataFrame) -> str:
    # Expect “Compare Dell XPS 13 and MacBook Air”
    match = re.search(
        r'compare\s+(.+?)\s+(?:and|with|vs)\s+(.+)', 
        user_request, re.IGNORECASE
    )
    if not match:
        return ("Please specify two laptops, e.g. “Compare Dell XPS 13 and MacBook Air.”")
    name1, name2 = match.group(1).strip(), match.group(2).strip()

    row1 = df[df['Laptop'].str.contains(name1, case=False)]
    row2 = df[df['Laptop'].str.contains(name2, case=False)]
    if row1.empty or row2.empty:
        return "One or both models weren’t found in our dataset."

    def specs(r):
        return {
            "CPU": r['CPU'],
            "RAM": f"{r['RAM']}GB",
            "Storage": f"{r['Storage']}GB {r['Storage type']}",
            "GPU": r['GPU'],
            "Screen": f"{r['Screen']}\" {'Touchscreen' if r['Touch']=='Yes' else 'Non-Touch'}",
            "Price": f"${r['Final Price']}"
        }
    s1, s2 = specs(row1.iloc[0]), specs(row2.iloc[0])

    lines = [f"**{row1.iloc[0]['Laptop']}** vs **{row2.iloc[0]['Laptop']}**"]
    for k in s1:
        lines.append(f"- **{k}:** {s1[k]}  |  {s2[k]}")
    return "\n".join(lines)


def handle_pick_one(user_request: str, df: pd.DataFrame) -> str:
    # Bước 1: Trích xuất tên laptop và mục đích sử dụng
    use_case = "general"
    patterns = [
        r'between\s+(.+?)\s+and\s+(.+?)\s+(?:for|with|in)\s+(.+?)(?:\?|$)',
        r'between\s+(.+?)\s+and\s+(.+?)(?:\?|$)',
        r'\b(?:pick|choose|select|recommend)\s+one\s+(.+?)\s+or\s+(.+?)\s+(?:for|with|in)\s+(.+?)(?:\?|$)',
        r'\b(?:pick|choose|select|recommend)\s+one\s+(.+?)\s+or\s+(.+?)(?:\?|$)'
    ]

    match = None
    for pattern in patterns:
        match = re.search(pattern, user_request, re.IGNORECASE)
        if match:
            break

    if not match:
        return "Please specify two laptops to compare, e.g. 'Which one between Dell XPS and MacBook Air for programming?'"

    groups = match.groups()
    laptop1_name = groups[0].strip()
    laptop2_name = groups[1].strip()

    if len(groups) >= 3:
        use_case = groups[2].strip().lower()

    # Bước 2: Làm sạch tên laptop
    def clean_laptop_name(name):
        stop_words = {'for', 'with', 'and', 'or', 'when', 'use', 'using',
                      'purpose', 'learning', 'code', 'tasks', 'task', 'like'}
        words = [word for word in name.split() if word.lower() not in stop_words]
        return " ".join(words).strip()

    laptop1_cleaned = clean_laptop_name(laptop1_name)
    laptop2_cleaned = clean_laptop_name(laptop2_name)

    # Bước 3: Tìm laptop trong dataset
    laptop1 = df[df['Laptop'].str.contains(laptop1_cleaned, case=False, na=False)]
    laptop2 = df[df['Laptop'].str.contains(laptop2_cleaned, case=False, na=False)]

    # Xử lý trường hợp không tìm thấy
    if laptop1.empty or laptop2.empty:
        missing = []
        if laptop1.empty:
            missing.append(f"'{laptop1_name}' (searched as: '{laptop1_cleaned}')")
        if laptop2.empty:
            missing.append(f"'{laptop2_name}' (searched as: '{laptop2_cleaned}')")

        suggestions = []
        for name in [laptop1_cleaned, laptop2_cleaned]:
            if name:
                similar = df[df['Laptop'].str.contains(name.split()[0], case=False)]['Laptop'].unique()
                if len(similar) > 0:
                    suggestions.append(f"Possible matches for '{name}': {', '.join(similar[:3])}")

        return (
                f"🔍 Couldn't find: {', '.join(missing)}.\n"
                "Please check model names or try simpler names.\n"
                + "\n".join(suggestions)
        )

    laptop1 = laptop1.iloc[0]
    laptop2 = laptop2.iloc[0]

    # Bước 4: Xác định mục đích sử dụng từ từ khóa
    use_cases = {
        'programming': ['programming', 'coding', 'developer', 'code', 'learn code'],
        'gaming': ['game', 'gaming'],
        'student': ['student', 'school', 'college'],
        'design': ['design', 'graphic', 'photoshop', 'illustrator'],
        'office': ['office', 'word', 'excel', 'powerpoint'],
        'ai': ['ai', 'machine learning', 'deep learning']
    }

    for case_key, keywords in use_cases.items():
        if any(keyword in user_request.lower() for keyword in keywords):
            use_case = case_key
            break

    # Bước 5: So sánh chi tiết các thông số
    # Hàm so sánh CPU
    def compare_cpu(cpu1, cpu2):
        cpu_rank = {'i3': 1, 'i5': 2, 'i7': 3, 'i9': 4, 'Ryzen 3': 1, 'Ryzen 5': 2, 'Ryzen 7': 3, 'Ryzen 9': 4}
        score1 = max((cpu_rank[chip] for chip in cpu_rank if chip in cpu1), default=0)
        score2 = max((cpu_rank[chip] for chip in cpu_rank if chip in cpu2), default=0)

        if score1 > score2 + 1:
            return f"{laptop1['Laptop']} has significantly better CPU"
        elif score1 > score2:
            return f"{laptop1['Laptop']} has better CPU"
        elif score2 > score1 + 1:
            return f"{laptop2['Laptop']} has significantly better CPU"
        elif score2 > score1:
            return f"{laptop2['Laptop']} has better CPU"
        return "Similar CPU performance"

    cpu_verdict = compare_cpu(laptop1['CPU'], laptop2['CPU'])

    # RAM
    ram1 = laptop1['RAM']
    ram2 = laptop2['RAM']
    ram_verdict = "Same RAM capacity"
    if ram1 > ram2:
        ram_verdict = f"{laptop1['Laptop']} has more RAM ({ram1}GB vs {ram2}GB)"
    elif ram2 > ram1:
        ram_verdict = f"{laptop2['Laptop']} has more RAM ({ram2}GB vs {ram1}GB)"

    # Giá
    price1 = laptop1['Final Price']
    price2 = laptop2['Final Price']
    price_verdict = "Similar price"
    if price1 < price2:
        price_verdict = f"{laptop1['Laptop']} is cheaper (${price1} vs ${price2})"
    elif price2 < price1:
        price_verdict = f"{laptop2['Laptop']} is cheaper (${price2} vs ${price1})"

    # GPU
    def gpu_score(gpu):
        if 'RTX 40' in gpu: return 5
        if 'RTX 30' in gpu: return 4
        if 'RTX 20' in gpu: return 3
        if 'GTX' in gpu: return 2
        return 1  # Integrated graphics

    gpu_score1 = gpu_score(laptop1['GPU'])
    gpu_score2 = gpu_score(laptop2['GPU'])
    gpu_verdict = "Similar GPU performance"
    if gpu_score1 > gpu_score2:
        gpu_verdict = f"{laptop1['Laptop']} has better GPU"
    elif gpu_score2 > gpu_score1:
        gpu_verdict = f"{laptop2['Laptop']} has better GPU"

    # Storage
    storage1 = laptop1['Storage']
    storage2 = laptop2['Storage']
    storage_type1 = laptop1['Storage type']
    storage_type2 = laptop2['Storage type']
    storage_verdict = "Similar storage"
    if storage1 > storage2:
        storage_verdict = f"{laptop1['Laptop']} has larger storage ({storage1}GB vs {storage2}GB)"
    elif storage2 > storage1:
        storage_verdict = f"{laptop2['Laptop']} has larger storage ({storage2}GB vs {storage1}GB)"
    elif 'NVMe' in storage_type1 and 'SATA' in storage_type2:
        storage_verdict = f"{laptop1['Laptop']} has faster NVMe SSD"
    elif 'NVMe' in storage_type2 and 'SATA' in storage_type1:
        storage_verdict = f"{laptop2['Laptop']} has faster NVMe SSD"

    # Màn hình
    screen1 = laptop1['Screen']
    screen2 = laptop2['Screen']
    screen_verdict = "Similar screen size"
    if screen1 > screen2:
        screen_verdict = f"{laptop1['Laptop']} has larger display ({screen1}\" vs {screen2}\")"
    elif screen2 > screen1:
        screen_verdict = f"{laptop2['Laptop']} has larger display ({screen2}\" vs {screen1}\")"

    # Pin (nếu có)
    battery_verdict = ""
    if 'Battery' in df.columns:
        bat1 = str(laptop1['Battery'])
        bat2 = str(laptop2['Battery'])
        wh1 = re.search(r'(\d+)\s*Wh', bat1)
        wh2 = re.search(r'(\d+)\s*Wh', bat2)
        if wh1 and wh2:
            wh1 = int(wh1.group(1))
            wh2 = int(wh2.group(1))
            if wh1 > wh2:
                battery_verdict = f"{laptop1['Laptop']} has better battery life"
            elif wh2 > wh1:
                battery_verdict = f"{laptop2['Laptop']} has better battery life"

    # Bước 6: Hệ thống tính điểm theo mục đích
    def calculate_score(laptop, use_case):
        score = 0

        # Programming weights
        if use_case == "programming":
            if 'i7' in laptop['CPU'] or 'i9' in laptop['CPU'] or 'Ryzen 7' in laptop['CPU'] or 'Ryzen 9' in laptop[
                'CPU']:
                score += 3
            if laptop['RAM'] >= 16:
                score += 4
            elif laptop['RAM'] >= 8:
                score += 2
            if 'NVMe' in laptop['Storage type']:
                score += 3
            elif 'SSD' in laptop['Storage type']:
                score += 2
            if laptop['Screen'] >= 15: score += 2
            if battery_verdict.startswith(laptop['Laptop']): score += 2

        # Gaming weights
        elif use_case == "gaming":
            if gpu_score(laptop['GPU']) >= 3: score += 5
            if 'i7' in laptop['CPU'] or 'i9' in laptop['CPU'] or 'Ryzen 7' in laptop['CPU'] or 'Ryzen 9' in laptop[
                'CPU']:
                score += 3
            if laptop['RAM'] >= 16:
                score += 4
            elif laptop['RAM'] >= 8:
                score += 2
            if laptop['Screen'] > 15: score += 2

        # General weights
        else:
            if 'i7' in laptop['CPU'] or 'i9' in laptop['CPU'] or 'Ryzen 7' in laptop['CPU'] or 'Ryzen 9' in laptop[
                'CPU']:
                score += 3
            if laptop['RAM'] >= 16:
                score += 3
            elif laptop['RAM'] >= 8:
                score += 2
            if 'NVMe' in laptop['Storage type']:
                score += 2
            elif 'SSD' in laptop['Storage type']:
                score += 1
            score += (1000 - laptop['Final Price']) / 100  # Ưu tiên giá thấp

        return round(score, 1)

    score1 = calculate_score(laptop1, use_case)
    score2 = calculate_score(laptop2, use_case)

    # Bước 7: Đưa ra khuyến nghị
    recommendation = None
    reasoning = []

    if score1 > score2:
        recommendation = laptop1['Laptop']
        reasoning.append(f"Higher {use_case} suitability score ({score1} vs {score2})")
    elif score2 > score1:
        recommendation = laptop2['Laptop']
        reasoning.append(f"Higher {use_case} suitability score ({score2} vs {score1})")
    else:
        recommendation = laptop1['Laptop']
        reasoning.append("Similar scores, first option selected")

    # Bước 8: Định dạng kết quả
    result = [
        f"🔹 **Comparison for {use_case.upper()} USE**",
        f"**{laptop1['Laptop']}** vs **{laptop2['Laptop']}**",
        "",
        "**Key Specifications:**",
        f"| Category | {laptop1['Laptop'][:20]} | {laptop2['Laptop'][:20]} |",
        "|----------|----------------------|----------------------|",
        f"| **CPU** | {laptop1['CPU']} | {laptop2['CPU']} |",
        f"| **RAM** | {laptop1['RAM']}GB | {laptop2['RAM']}GB |",
        f"| **Storage** | {laptop1['Storage']}GB {laptop1['Storage type']} | {laptop2['Storage']}GB {laptop2['Storage type']} |",
        f"| **GPU** | {laptop1['GPU'][:20]} | {laptop2['GPU'][:20]} |",
        f"| **Screen** | {laptop1['Screen']}\" | {laptop2['Screen']}\" |",
        f"| **Price** | ${laptop1['Final Price']} | ${laptop2['Final Price']} |",
        "",
        "**Key Differences Analysis:**",
        f"- CPU: {cpu_verdict}",
        f"- RAM: {ram_verdict}",
        f"- Storage: {storage_verdict}",
        f"- GPU: {gpu_verdict}",
        f"- Screen: {screen_verdict}",
        f"- Price: {price_verdict}"
    ]

    if battery_verdict:
        result.append(f"- Battery: {battery_verdict}")

    result.extend([
        "",
        f"**{use_case.capitalize()} Suitability Score:**",
        f"- {laptop1['Laptop'][:25]}: {score1}/10",
        f"- {laptop2['Laptop'][:25]}: {score2}/10",
        "",
        "🚀 **Recommendation:**",
        f"👉 **{recommendation}**",
        f"**Reason:** {'; '.join(reasoning)}"
    ])

    return "\n".join(result)

import streamlit as st
from invoice import insert_invoice

def handle_buy_laptop(user_request: str):
    """Hiển thị form và lưu đơn hàng sau khi submit"""
    st.write(f"🎉 Awesome! You want to order laptop: **{user_request}**")
    st.write("Please enter below information to complete order process:")

    with st.form("buy_laptop_form"):
        full_name = st.text_input("Your fullname")
        phone = st.text_input("Your phone number")
        address = st.text_area("your address")

        col1, col2 = st.columns(2)
        with col1:
            submitted = st.form_submit_button("📦 Xác nhận mua")
        with col2:
            canceled = st.form_submit_button("❌ Hủy bỏ")

        if canceled:
            st.session_state.show_buy_form = False

        if submitted:
            if not full_name or not phone or not address:
                st.warning("⛔ Vui lòng điền đầy đủ thông tin trước khi xác nhận.")
            else:
                insert_invoice(full_name, user_request, phone, address)
                st.success("✅ Đã nhận thông tin đặt hàng. Cảm ơn bạn! Chúng tôi sẽ liên hệ và giao hàng trong 1–3 ngày.")
                st.write("**📄 Thông tin đơn hàng:**")
                st.markdown(f"- **Máy:** {user_request}")
                st.markdown(f"- **Tên:** {full_name}")
                st.markdown(f"- **SĐT:** {phone}")
                st.markdown(f"- **Địa chỉ:** {address}")
                st.session_state.show_buy_form = False  # Ẩn form
