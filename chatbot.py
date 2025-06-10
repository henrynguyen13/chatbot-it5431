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

from intents import detect_intent, handle_intent

# --- Setup Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Load Environment Variables ---
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    raise ValueError("TOGETHER_API_KEY not found in environment variables")

# ======================= 1. MODEL & SYSTEM CONFIGURATION (ENGLISH) =======================
# Use a high-quality English embedding model. 'all-MiniLM-L6-v2' is a standard, efficient choice.
embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

# Use the specified Llama 3 model, which is excellent for English and reasoning.
llm = ChatTogether(
    # The identifier for Llama 3 on Together.ai is typically this format
    model="meta-llama/Llama-3-8B-chat-hf", 
    temperature=0.2,  # Lower temperature for more factual and less creative answers
    max_tokens=1024,
    api_key=api_key
)

# --- Load and Clean Data ---
df = pd.read_csv("data/laptops.csv")
storage_type_mode = df['Storage type'].mode()[0] if not df['Storage type'].mode().empty else "Unknown"
df['Storage type'].fillna(storage_type_mode, inplace=True)
df['GPU'].fillna("None", inplace=True)
df['Screen'].fillna(round(df['Screen'].mean(), 1), inplace=True)
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
    
    documents, metadatas, ids = [], [], []

    for idx, row in df.iterrows():
        # --- Create natural English sentences for semantic search ---
        touch_text = "a touchscreen" if row['Touch'] == 'Yes' else "a non-touch screen"
        description = (
            f"This is the {row['Laptop']}, a {row['Status']} model from the brand {row['Brand']}. "
            f"It is equipped with an {row['CPU']} processor, {row['RAM']}GB of RAM, and {row['Storage']}GB of {row['Storage type']} storage. "
            f"It features a {row['GPU']} GPU and a {row['Screen']}-inch {touch_text}."
        )
        documents.append(description)
        
        # --- Store structured data in metadata for precise filtering and display ---
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
            "touchscreen": str(row['Touch']), # 'Yes' or 'No'
            "price_usd": float(row['Final Price']),
            "full_description": description
        }
        metadatas.append(metadata)
        ids.append(str(idx))
    
    # Batch add to ChromaDB
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

# ======================= 2. PROMPT TEMPLATE (ENGLISH) =======================
prompt_template = PromptTemplate(
    input_variables=["user_request", "context"],
    template="""
You are a top-tier, professional, and friendly laptop sales assistant. Your primary goal is to provide clear, well-formatted, and helpful recommendations.

**Customer's Request:** "{user_request}"

Based **only** on the list of available laptops provided in the context, present all suitable products that are a good match for the customer's request.

---
**CRITICAL INSTRUCTIONS:**

1.  **Format Your Response EXACTLY like the example below.** Use Markdown for bolding and bullet points. Each specification must be on a new line.
2.  **Be Concise:** Start the response directly with the recommendations. Do not add introductory sentences like "Based on your request...".
3.  **Strictly Adhere to Data:** Only use information from the "Available Laptops" section. Do not invent specs or products.
4.  **Justify Your Choice:** Include a short, one-sentence "Reasoning:" that explains why the laptop is a good match.
5.  **Handle Empty Context:** If the "Available Laptops" section is empty, respond with **only** this exact sentence: "Unfortunately, I couldn't find any laptops that perfectly match all your criteria in our current inventory. Would you like to try a broader search?"

---
**EXAMPLE OF THE REQUIRED OUTPUT FORMAT:**

Here are the best matches I found for you:

- **HP Spectre x360 14**
  - **Status:** New
  - **CPU:** Intel Core i7-1355U
  - **RAM:** 16GB
  - **Storage:** 1TB SSD
  - **GPU:** Intel Iris Xe
  - **Screen:** 13.5" Touchscreen
  - **Price:** $1249.99
  - **Reasoning:** This laptop perfectly matches your request for a lightweight touchscreen model around $1200.

- **Dell XPS 15**
  - **Status:** New
  - **CPU:** Intel Core i9-13900H
  - **RAM:** 32GB
  - **Storage:** 1TB SSD
  - **GPU:** NVIDIA GeForce RTX 4070
  - **Screen:** 15.6" Non-Touch
  - **Price:** $2199.00
  - **Reasoning:** With its powerful i9 CPU and dedicated RTX 4070 graphics card, this is an excellent choice for your video editing needs.

---
**Available Laptops:**
{context}
"""
)
chain = prompt_template | llm

#  3. RETRIEVE-THEN-FILTER LOGIC (ENGLISH)
def parse_query_to_filters(user_request: str) -> dict:
    """Extracts structured filters from a natural language query using regex."""
    filters = {}
    
    # Brands
    brands = ['dell', 'hp', 'asus', 'lenovo', 'msi', 'acer', 'apple', 'macbook']
    for brand in brands:
        if re.search(r'\b' + brand + r'\b', user_request, re.IGNORECASE):
            filters['brand'] = 'APPLE' if brand in ['apple', 'macbook'] else brand.upper()
            break

    # RAM
    ram_match = re.search(r'(\d+)\s*gb\s*ram', user_request, re.IGNORECASE)
    if ram_match:
        filters['ram_gb'] = int(ram_match.group(1))

    # Price
    price_match = re.search(r'(under|below|less than|over|above|more than|around)\s*\$?(\d+)', user_request, re.IGNORECASE)
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
            
    # Touchscreen
    if re.search(r'touch\s*screen|touchscreen', user_request, re.IGNORECASE):
        filters['touchscreen'] = 'Yes'

    return filters

def filter_laptops(user_request: str) -> str:
    """Finds laptops using the Retrieve-then-Filter strategy."""
    
    # --- Step 1: Retrieve a broad set of candidates ---
    try:
        # Retrieve more results (e.g., 25) to have a larger pool for filtering
        results = collection.query(query_texts=[user_request], n_results=25, include=["metadatas"])
    except Exception as e:
        logger.error(f"Error during retrieval: {e}")
        return "An error occurred while searching for laptops. Please try again later."

    if not results or not results.get('metadatas') or not results['metadatas'][0]:
        return "I'm sorry, I couldn't find any relevant laptops for your request."

    # --- Step 2: Parse query and apply precise filters ---
    filters = parse_query_to_filters(user_request)
    logger.info(f"Parsed filters from query: {filters}")

    retrieved_metadatas = results['metadatas'][0]
    filtered_laptops = []
    
    for meta in retrieved_metadatas:
        is_match = True
        # Check each filter against the metadata
        if filters.get('brand') and meta.get('brand', '').upper() != filters['brand']:
            is_match = False
        if filters.get('ram_gb') and meta.get('ram_gb') != filters['ram_gb']:
            is_match = False
        if filters.get('max_price_usd') and meta.get('price_usd') > filters['max_price_usd']:
            is_match = False
        if filters.get('min_price_usd') and meta.get('price_usd') < filters['min_price_usd']:
            is_match = False
        if filters.get('touchscreen') and meta.get('touchscreen') != filters['touchscreen']:
            is_match = False
            
        if is_match:
            filtered_laptops.append(meta)

    # --- Step 3: Prepare context and generate response ---
    if not filtered_laptops:
        return "Unfortunately, I couldn't find any laptops that perfectly match all your criteria (like price, brand, or RAM). Would you like to try a broader search?"
    
    # Create a clean context for the LLM from the truly matching laptops
    # Limit to the top 5 matches to avoid overwhelming the LLM
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
    context = "\n".join(context_list)
    
    logger.info("--- Clean Context Sent to LLM ---")
    logger.info(context)
    logger.info("---------------------------------")
    
    try:
        response = chain.invoke({"user_request": user_request, "context": context})
        return response.content.strip()
    except Exception as e:
        logger.error(f"Error during LLM generation: {e}")
        return f"An error occurred while generating the response: {e}"


def process_query(
        user_request: str,
) -> str:
    """Xử lý truy vấn với context lịch sử"""
    # Kiểm tra intent
    intent = detect_intent(user_request)

    # Xử lý intent đặc biệt
    if intent != "search":
        return handle_intent(intent, user_request), []

    # Xử lý truy vấn tìm kiếm thông thường
    return filter_laptops(user_request)

# Khởi tạo manager
# history_manager = ChatHistoryManager()