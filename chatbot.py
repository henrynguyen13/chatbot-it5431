import os
import pandas as pd
from dotenv import load_dotenv
import json
from langchain_together import ChatTogether
from langchain.prompts import PromptTemplate
import chromadb
from chromadb.utils.embedding_functions import TogetherAIEmbeddingFunction
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
import time
import logging
import hashlib

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
api_key = os.getenv("TOGETHER_API_KEY")
if not api_key:
    raise ValueError("TOGETHER_API_KEY not found in environment variables")
os.environ["CHROMA_TOGETHER_AI_API_KEY"] = api_key

#Initialize TogetherAI embedding function
embedding_function = TogetherAIEmbeddingFunction(
    model_name="togethercomputer/m2-bert-80M-32k-retrieval",
)

# Load laptop data
df = pd.read_csv("data/laptops.csv")

storage_type_mode = df['Storage type'].mode()[0] if not df['Storage type'].mode().empty else "Unknown"
df['Storage type'].fillna(storage_type_mode, inplace=True)

# Fill GPU with "None"
df['GPU'].fillna("None", inplace=True)
# Fill Screen with the mean value
df['Screen'].fillna(df['Screen'].mean(), inplace=True)
# Verify after filling
missing_values_after = df.isna().sum()
logger.info("Missing values after filling:\n%s", missing_values_after)

# Initialize Chroma vector database with persistence
client = chromadb.PersistentClient(path="./chroma_data")
collection_name = "laptops"

# Hàm tạo hash từ dữ liệu để kiểm tra thay đổi
def compute_data_hash(df):
    # Tạo chuỗi đại diện cho dữ liệu
    data_str = df.to_csv(index=False)
    return hashlib.md5(data_str.encode()).hexdigest()

# Hàm kiểm tra và lưu hash
def save_data_hash(data_hash, file_path="data_hash.txt"):
    with open(file_path, "w") as f:
        f.write(data_hash)

def load_data_hash(file_path="data_hash.txt"):
    try:
        with open(file_path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return None

# Kiểm tra collection đã tồn tại và dữ liệu có thay đổi không
def initialize_collection():
    data_hash = compute_data_hash(df)
    stored_hash = load_data_hash()
    
    try:
        collection = client.get_collection(collection_name, embedding_function=embedding_function)
        logger.info("Found existing collection: %s", collection_name)
        
        # Kiểm tra xem dữ liệu có thay đổi không
        if stored_hash == data_hash:
            logger.info("Data unchanged, using existing collection with %d documents", collection.count())
            return collection
        else:
            logger.info("Data changed, deleting old collection and creating new one")
            client.delete_collection(collection_name)
    except:
        logger.info("No existing collection found, creating new one")
    
    # Tạo collection mới
    collection = client.create_collection(collection_name, embedding_function=embedding_function)
    
    # Chuẩn bị documents và IDs
    documents = []
    ids = []
    for idx, row in df.iterrows():
        price_vnd = row['Final Price'] * 25000  # Giả sử 1 USD = 25,000 VND
        doc = (
            f"Laptop: {row['Laptop']}, "
            f"Status: {row['Status']}, "
            f"Brand: {row['Brand']}, "
            f"Model: {row['Model']}, "
            f"CPU: {row['CPU']}, "
            f"RAM: {row['RAM']}GB, "
            f"Storage: {row['Storage']}GB {row['Storage type']}, "
            f"GPU: {row['GPU'] if pd.notna(row['GPU']) else 'None'}, "
            f"Screen: {row['Screen']} inches, "
            f"Touch: {'Yes' if row['Touch'] == 'Yes' else 'No'}, "
            f"Price: {price_vnd} VND"
        )
        documents.append(doc)
        ids.append(str(idx))  # Sử dụng index của DataFrame làm ID
    
    # Thêm dữ liệu vào Chroma với batch size lớn hơn
    batch_size = 5  # Tăng batch size để xử lý nhanh hơn
    max_retries = 5
    for i in range(0, len(documents), batch_size):
        batch_docs = documents[i:i + batch_size]
        batch_ids = ids[i:i + batch_size]
        for attempt in range(max_retries):
            try:
                collection.add(documents=batch_docs, ids=batch_ids)
                logger.info("Added batch %d with %d documents", i // batch_size + 1, len(batch_docs))
                break
            except Exception as e:
                logger.error("Error adding batch %d (attempt %d): %s", i // batch_size + 1, attempt + 1, e)
                if attempt < max_retries - 1:
                    wait_time = (2 ** attempt) * 5
                    logger.info("Retrying after %d seconds...", wait_time)
                    time.sleep(wait_time)
                else:
                    logger.error("Failed to add batch %d after %d attempts", i // batch_size + 1, max_retries)
                    raise
    
    # Lưu hash của dữ liệu
    save_data_hash(data_hash)
    logger.info("New collection created and data hash saved")
    return collection

# Khởi tạo collection
collection = initialize_collection()

# Initialize TogetherAI LLM
llm = ChatTogether(
    model="mistralai/Mistral-7B-Instruct-v0.1",
    temperature=0.5,
    max_tokens=1024,
    api_key=api_key
)

# Prompt template for RAG
prompt_template = PromptTemplate(
    input_variables=["user_input", "context"],
    template="""
Bạn là trợ lý bán hàng laptop chuyên nghiệp. Người dùng yêu cầu: "{user_input}"

Dựa trên thông tin laptop sau, hãy đề xuất 1-3 laptop phù hợp nhất, đảm bảo:
- Khớp chính xác thương hiệu, giá (1 USD = 25,000 VND), RAM, CPU, hoặc các thông số khác nếu người dùng đề cập.
- Hiển thị thông tin: thương hiệu, model, CPU, RAM, bộ nhớ, GPU, màn hình, cảm ứng, và giá (khoảng X VND (Y USD)).
- Nếu không có laptop nào phù hợp, trả lời: "Không tìm thấy laptop phù hợp với yêu cầu của bạn."
- Trả lời bằng tiếng Việt chuẩn, tự nhiên, đúng ngữ pháp, không dùng từ sai chính tả hoặc thuật ngữ tiếng Anh.

Thông tin laptop:
{context}
"""
)
chain = prompt_template | llm


def filter_laptops(user_input):
    # Retrieve top 5 relevant laptops from Chroma with retry
    max_retries = 5
    for attempt in range(max_retries):
        try:
            results = collection.query(query_texts=[user_input], n_results=5)
            context = "\n".join(results['documents'][0])
            if not context:
                return "Không tìm thấy laptop phù hợp với yêu cầu của bạn."
            break
        except Exception as e:
            logger.error("Error during retrieval (attempt %d): %s", attempt + 1, e)
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 5
                logger.info("Retrying retrieval after %d seconds...", wait_time)
                time.sleep(wait_time)
            else:
                return f"Error during retrieval after {max_retries} attempts: {e}"
    
    # Generate response using LLM
    try:
        response = chain.invoke({"user_input": user_input, "context": context})
        return response.content.strip()
    except Exception as e:
        logger.error("Error during LLM generation: %s", e)
        return f"Error generating response: {e}"