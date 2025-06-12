import hashlib
import logging
import os
import re
import chromadb
import pandas as pd
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_together import ChatTogether
from langchain.memory import ConversationBufferMemory
from intents import detect_intent, handle_intent, handle_compare, handle_pick_one, handle_buy_laptop

#LCEL

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    raise ValueError("TOGETHER_API_KEY not found in environment variables")

# --- Khởi tạo ConversationBufferMemory ---
memory = ConversationBufferMemory(return_messages=True, input_key="user_request", output_key="response")

# --- Model & System Configuration ---
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
llm = ChatTogether(
    model="meta-llama/Llama-3-8B-chat-hf",
    temperature=0.3,
    max_tokens=1024,
    api_key=api_key
)

# --- Load and Clean Data ---
df = pd.read_csv("data/laptops.csv")
storage_type_mode = df['Storage type'].mode()[0] if not df['Storage type'].mode().empty else "Unknown"
df['Storage type'].fillna(storage_type_mode, inplace=True)
df['GPU'].fillna("None", inplace=True)
df['Screen'].fillna(round(df['Screen'].mean(), 1), inplace=True)
df['Brand'] = df['Brand'].str.upper()
logger.info("Data loaded and cleaned successfully.")

# --- ChromaDB Setup ---
client = chromadb.PersistentClient(path="./chroma_data_en")
collection_name = "laptops_en"

def compute_data_hash(df):
    data_str = df.to_csv(index=False)
    return hashlib.md5(data_str.encode()).hexdigest()

def save_data_hash(data_hash, file_path="data_hash_en.txt"):
    with open(file_path, "w") as f:
        f.write(data_hash)

def load_data_hash(file_path="data_hash_en.txt"):
    try:
        with open(file_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

def initialize_collection():
    data_hash = compute_data_hash(df)
    stored_hash = load_data_hash()
    
    try:
        collection = client.get_collection(collection_name, embedding_function=embedding_function)
        logger.info(f"Found existing collection: '{collection_name}'")
        if stored_hash == data_hash:
            logger.info(f"Data is unchanged. Using existing collection with {collection.count()} documents.")
            return collection
        else:
            logger.info("Data has changed. Deleting old collection and recreating.")
            client.delete_collection(collection_name)
    except Exception:
        logger.info("No existing collection found. Creating a new one.")
    
    collection = client.create_collection(collection_name, embedding_function=embedding_function)
    #combine fulltext search and embedding search 
    documents, metadatas, ids = [], [], []
    for idx, row in df.iterrows():
        touch_text = "a touchscreen" if row['Touch'] == 'Yes' else "a non-touch screen"
        description = (
            f"This is the {row['Laptop']}, a {row['Status']} model from the brand {row['Brand']}. "
            f"It is equipped with an {row['CPU']} processor, {row['RAM']}GB of RAM, and {row['Storage']}GB of {row['Storage type']} storage. "
            f"It features a {row['GPU']} GPU and a {row['Screen']}-inch {touch_text}."
        )
        documents.append(description)
        metadata = {
            "laptop_name": str(row['Laptop']),
            "status": str(row['Status']),
            "brand": str(row['Brand']),
            "model": str(row['Model']),
            "cpu": str(row['CPU']),
            "ram_gb": int(row['RAM']),
            "storage_gb": int(row['Storage']),
            "storage_type": str(row['Storage type']),
            "gpu": str(row['GPU']),
            "screen_inches": float(row['Screen']),
            "touchscreen": str(row['Touch']),
            "price_usd": float(row['Final Price']),
            "full_description": description
        }
        metadatas.append(metadata)
        ids.append(str(idx))
    
    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i + batch_size],
            metadatas=metadatas[i:i + batch_size],
            ids=ids[i:i + batch_size]
        )
        logger.info(f"Added batch {i // batch_size + 1} with {len(documents[i:i + batch_size])} documents.")
    
    save_data_hash(data_hash)
    logger.info("New collection created and data hash saved.")
    return collection

collection = initialize_collection()

#Compare 2 types of laptop
#Get order key -> change the prompt based on query (do not list,..)
#Change the temperature 
#System prompt +  user prompt -> change the user prompt 

# --- Prompt Template ---
prompt_template = PromptTemplate(
    input_variables=["user_request", "context", "history"],
    template="""
You are a top-tier, professional, and friendly laptop sales assistant. Your primary goal is to provide clear, well-formatted, and helpful recommendations.

**Customer's Request:** "{user_request}"

**Conversation History: (focus on recent interactions to understand context)**
{history}
**Available Laptops:**  
{context}

---

**GENERAL INSTRUCTIONS:**

1. Always respond using **Markdown** format.
2. Only use laptops listed in the "Available Laptops" section. **Never make up specs or products**.
3. Keep the tone concise, helpful, and professional.
4. If the context is empty, respond exactly with:  
   "Unfortunately, I couldn't find any laptops that perfectly match all your criteria in our current inventory. Would you like to try a broader search?"
---

**HANDLE THE FOLLOWING SCENARIOS:**

### 🔹 1. Recommendation (Default)
- If the customer asks for a suggestion, list all matching laptops.
- Follow this format:

Here are the best matches I found for you:

- **Laptop Name**
  - **Status:** New/Used
  - **CPU:** ...
  - **RAM:** ...
  - **Storage:** ...
  - **GPU:** ...
  - **Screen:** ...
  - **Price:** ...
  - **Reasoning:** Short explanation why it matches

---

### 🔹 2. Comparison (e.g. "Compare X and Y")
- Format the output as a **Markdown table** comparing specs side-by-side:

| Spec     | Laptop A         | Laptop B         |
|----------|------------------|------------------|
| CPU      | Intel Core i5... | Intel Core i7... |
| RAM      | 8GB              | 16GB             |
| Storage  | 256GB SSD        | 512GB SSD        |
| GPU      | Intel Iris Xe    | NVIDIA MX450     |
| Price    | $899.00          | $1099.00         |
| Reasoning| A is cheaper...  | B is more powerful... |

---

### 🔹 3. Pick One (e.g. "Which one should I choose?")
- **STRICTLY CHOOSE ONLY ONE LAPTOP FROM THE CONTEXT BELOW**
- **NEVER CHOOSE A LAPTOP NOT LISTED IN THE CONTEXT**
- Compare the top 2 options in the context and select the best one
- Format:

After careful comparison, my top recommendation is **[Laptop Name EXACTLY AS IN CONTEXT]** because:

- **CPU:** ...
- **RAM:** ...
- **Storage:** ...
- **Price:** ...
- **Reasoning:** Short comparison highlighting why this is better than the alternative

**IMPORTANT:**
1. You MUST select only ONE laptop
2. The chosen laptop MUST be from the context
3. Never suggest unlisted laptops

---

### 🔹 4. Order or Rank (e.g. "Sort by price/RAM")
- Return a **sorted list** of matching laptops ordered by the requested criteria.
- Format:

Here are the laptops sorted by [criteria]:

1. **Laptop A**
   - **Price:** $799
   - **RAM:** 8GB
   - ...

2. **Laptop B**
   - **Price:** $999
   - **RAM:** 16GB
   - ...

---

### 🔹 5. Filter Further (e.g. "Which one is lighter?", "Which one has better battery?")
- Apply an extra filter based on the user's clarification or follow-up and return refined results.

---

### 🔹 6. Not Found Case
If no matching laptops are found, return:

"Unfortunately, I couldn't find any laptops that perfectly match all your criteria in our current inventory. Would you like to try a broader search?"
"""
)

chain = prompt_template | llm


# --- Retrieve-then-Filter Logic ---
def parse_query_to_filters(user_request: str, memory: ConversationBufferMemory, filter_history: list) -> dict:
    """
    Parse the user's query to extract laptop search filters (USD only).
    Incorporate context from conversation history if the query is ambiguous.
    """
    filters = filter_history[-1].copy() if filter_history else {}
    brands = ['dell', 'hp', 'asus', 'lenovo', 'msi', 'acer', 'apple', 'macbook']

    # Load conversation history
    # history = memory.load_memory_variables({})["history"]
    
    # Handle context-dependent queries (e.g., "cheaper", "less expensive")
    if filter_history and any(keyword in user_request.lower() for keyword in ["cheaper", "less expensive", "more affordable"]):
        # Safely retrieve filters from the previous message's additional_kwargs
        # last_additional_kwargs = history[-2].additional_kwargs if len(history) >= 2 and hasattr(history[-2], 'additional_kwargs') else {}
        # last_filters = last_additional_kwargs.get("filters", {})
        last_filters = filter_history[-1]
        filters.update(last_filters)  # Retain previous filters
        if last_filters.get("max_price_usd"):
            filters["max_price_usd"] = last_filters["max_price_usd"] * 0.9  # Reduce price by 10%
    
    # Detect brand
    for brand in brands:
        if re.search(r'\b' + brand + r'\b', user_request, re.IGNORECASE):
            filters['brand'] = 'APPLE' if brand in ['apple', 'macbook'] else brand.upper()
            break
    
    # Detect RAM
    ram_match = re.search(r'(\d+)\s*gb\s*ram', user_request, re.IGNORECASE)
    if ram_match:
        filters['ram_gb'] = int(ram_match.group(1))
    
    # Detect price (USD only)
    price_match = re.search(
        r'(under|below|less than|over|above|more than|around)\s*\$?(\d+\.?\d*)',
        user_request, re.IGNORECASE
    )
    if price_match:
        limit_type, value = price_match.groups()
        price_value = float(value)
        if limit_type in ['under', 'below', 'less than']:
            filters['max_price_usd'] = price_value
        elif limit_type in ['over', 'above', 'more than']:
            filters['min_price_usd'] = price_value
        elif limit_type == 'around':
            filters['min_price_usd'] = price_value * 0.9
            filters['max_price_usd'] = price_value * 1.1
    
    # Detect touchscreen
    if re.search(r'touch\s*screen|touchscreen', user_request, re.IGNORECASE):
        filters['touchscreen'] = 'Yes'
    
    # Detect usage intent
    if "gaming" in user_request.lower():
        filters['gpu'] = ['NVIDIA', 'AMD']  # Require strong GPU
        filters['ram_gb'] = filters.get('ram_gb', 16)  # Default to 16GB RAM
        filters['screen_inches'] = filters.get('screen_inches', 15.6)  # Default to ≥15.6 inches
    elif "student" in user_request.lower() or "study" in user_request.lower():
        filters['max_price_usd'] = filters.get('max_price_usd', 500)  # Default to under $500
        filters['ram_gb'] = filters.get('ram_gb', 8)  # Default to 8GB RAM
    
    # Detect screen size
    screen_match = re.search(r'(\d+\.?\d*)\s*(inch|")', user_request, re.IGNORECASE)
    if screen_match:
        filters['screen_inches'] = float(screen_match.group(1))
    
    # Detect storage (SSD/HDD)
    storage_match = re.search(r'(\d+)\s*(gb|tb)\s*(ssd|hdd)', user_request, re.IGNORECASE)
    if storage_match:
        size, unit, storage_type = storage_match.groups()
        size = float(size) * (1000 if unit.lower() == 'tb' else 1)
        filters['storage_gb'] = size
        filters['storage_type'] = storage_type.upper()
    
    return filters
def filter_laptops(user_request: str, filters: dict) -> str:
    where_clause = {"$and": []} if len(filters) > 1 else {}

    if filters.get("brand"):
        where_clause["$and"].append({"brand": filters["brand"]}) if len(filters) > 1 else where_clause.update({"brand": filters["brand"]})
    if filters.get("ram_gb"):
        where_clause["$and"].append({"ram_gb": filters["ram_gb"]}) if len(filters) > 1 else where_clause.update({"ram_gb": filters["ram_gb"]})
    if filters.get("touchscreen"):
        where_clause["$and"].append({"touchscreen": filters["touchscreen"]}) if len(filters) > 1 else where_clause.update({"touchscreen": filters["touchscreen"]})
    if filters.get("storage_gb"):
        where_clause["$and"].append({"storage_gb": filters["storage_gb"]}) if len(filters) > 1 else where_clause.update({"storage_gb": filters["storage_gb"]})
    if filters.get("storage_type"):
        where_clause["$and"].append({"storage_type": filters["storage_type"]}) if len(filters) > 1 else where_clause.update({"storage_type": filters["storage_type"]})
    if filters.get("screen_inches"):
        where_clause["$and"].append({"screen_inches": filters["screen_inches"]}) if len(filters) > 1 else where_clause.update({"screen_inches": filters["screen_inches"]})

    if len(where_clause.get("$and", [])) == 0:
        where_clause = {} if not where_clause else where_clause
    elif len(where_clause["$and"]) == 1:
        where_clause = where_clause["$and"][0]

    print("where_clause:", where_clause)
    results = collection.query(
        query_texts=[user_request],
        n_results=50,  # Tăng số kết quả ban đầu để có nhiều lựa chọn hơn
        where=where_clause if where_clause else None,
        include=["metadatas"]
    )

    if not results or not results.get('metadatas') or not results['metadatas'][0]:
        return "I'm sorry, I couldn't find any relevant laptops for your request."
    
    filtered_laptops = []
    for meta in results['metadatas'][0]:
        is_match = True
        price_usd = meta.get('price_usd')
        
        if filters.get('max_price_usd') and price_usd > filters['max_price_usd']:
            is_match = False
        if filters.get('min_price_usd') and price_usd < filters['min_price_usd']:
            is_match = False
        if filters.get('gpu') and not any(gpu in meta.get('gpu', '').upper() for gpu in filters['gpu']):
            is_match = False
        if is_match:
            filtered_laptops.append(meta)
    
    if not filtered_laptops:
        return "Unfortunately, I couldn't find any laptops that perfectly match all your criteria. Would you like to try a broader search?"
    
    context_list = []
    for meta in filtered_laptops[:5]:
        context_list.append(
            f" - Product: {meta['laptop_name']}\n"
            f"  - Status: {meta['status']}\n"
            f"  - CPU: {meta['cpu']}\n"
            f"  - RAM: {meta['ram_gb']}GB\n"
            f"  - Storage: {meta['storage_gb']}GB {meta['storage_type']}\n"
            f"  - GPU: {meta['gpu']}\n"
            f"  - Screen: {meta['screen_inches']}\" {'Touchscreen' if meta['touchscreen'] == 'Yes' else 'Non-Touch'}\n"
            f"  - Price: ${meta['price_usd']:,.2f}"
        )
    return "\n".join(context_list)

def process_query(user_request: str, filter_history: list) -> str:
    intent = detect_intent(user_request)
    print("intent", intent)  # Debugging line to check detected intent
   
    if intent != "search" and intent not in ["compare", "pick_one", "sort", "filter_further"]:
        response = handle_intent(intent, user_request)
        memory.save_context({"user_request": user_request}, {"response": response})
        return response

    if intent == "pick_one":
        response = handle_pick_one(user_request, df)
        memory.save_context({"user_request": user_request}, {"response": response})
        return response

    if intent == "buy_laptop":
        # Trích xuất tên laptop từ request
        laptop_match = re.search(
            r"(asus|hp|lenovo|dell|msi|acer|apple|macbook|vivobook|ideapad|inspiron|swift|xps|thinkpad|surface)[\w\s-]*",
            user_request, re.IGNORECASE
        )
        laptop_name = laptop_match.group(0) if laptop_match else "laptop"

        # Trả về mã HTML/JS để kích hoạt form
        return f"""
           <script>
           setTimeout(() => {{
               document.getElementById('buy-laptop-btn').click();
               document.getElementById('laptop-name').value = '{laptop_name}';
           }}, 500);
           </script>
           """

    filters = parse_query_to_filters(user_request, memory, filter_history)  # Pass memory
    print("PARSE", filters)  # Debugging line to check parsed filters
    filter_history.append(filters)
    context = filter_laptops(user_request, filters)
    print("CONTEXT", context)  # Debugging line to check the context
    history = memory.load_memory_variables({})["history"]
    history_str = "\n".join([f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}" for msg in history[-6:]])
    
    response = chain.invoke({
        "user_request": user_request,
        "context": context,
        "history": history_str
    })
    # Save filters in additional_kwargs
    memory.save_context(
        {"user_request": user_request},
        {"response": response.content}
    )
    return response.content

def clear_memory():
    """Xóa bộ nhớ của ConversationBufferMemory khi bắt đầu phiên mới"""
    memory.clear()

def load_memory_from_history(chat_history: list):
    """Tải lịch sử từ chat_history vào ConversationBufferMemory"""
    memory.clear()  # Xóa bộ nhớ hiện tại
    for i in range(0, len(chat_history), 2):
        if i + 1 < len(chat_history):
            user_msg = chat_history[i]["message"] if chat_history[i]["role"] == "user" else None
            assistant_msg = chat_history[i+1]["message"] if chat_history[i+1]["role"] == "assistant" else None
            if user_msg and assistant_msg:
                memory.save_context({"user_request": user_msg}, {"response": assistant_msg})
