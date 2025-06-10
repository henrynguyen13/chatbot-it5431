import re


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
    has_spec_numbers = re.search(r'\b(\d+\s*(gb|gb ram|ram|ssd|hdd|tb|usd|vnd|\$))\b', query)

    # Phát hiện từ khóa liên quan đến laptop
    has_laptop_keyword = any(keyword in query for keyword in laptop_keywords)

    # Phát hiện cấu trúc câu hỏi tìm kiếm điển hình
    has_search_pattern = re.search(r'(laptop|notebook).*(for|with|under|around|that)', query)

    if has_spec_numbers or has_laptop_keyword or has_search_pattern:
        return "search"

    # Không xác định được intent
    return "unknown"


def handle_intent(intent: str, user_request: str = "") -> str:
    """Xử lý intent và trả về phản hồi phù hợp"""
    responses = {
        "greeting": "Hello! I'm your laptop shopping assistant. 🤖 How can I help you find the perfect laptop today?",
        "thanks": "You're welcome! 😊 If you need more help finding a laptop, just let me know!",
        "goodbye": "Goodbye! 👋 Feel free to come back if you need laptop recommendations anytime!",
        "self_introduction": "I'm a specialized AI assistant for laptop recommendations. My job is to help you find the perfect laptop based on your needs, budget, and preferences!",
        "capabilities": "I can help you:\n- Find laptops matching your budget 💰\n- Recommend laptops for specific uses (gaming, work, etc.) 🎮💼\n- Compare laptop specifications 📊\n- Suggest the best value laptops ⭐\n\nJust tell me what you're looking for!",
        "unknown": "I'm here to help you find the perfect laptop! 🚀 Could you tell me what kind of laptop you're looking for?"
    }

    return responses.get(intent, responses["unknown"])