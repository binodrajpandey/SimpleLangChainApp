from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Step 1: Create embeddings
embedding_model = OpenAIEmbeddings()
texts = ["Paris is the capital of France", "Tokyo is in Japan", "Soccer is popular in Europe"]

# Step 2: Store in a Vector Store
vectorstore = FAISS.from_texts(texts, embedding_model)

# Step 3: Query with meaning
query = "What is the capital of France?"
docs = vectorstore.similarity_search(query, k=2)

print(docs[0].page_content)  # → "Paris is the capital of France"
print(docs[1].page_content)  # → "Paris is the capital of France"
