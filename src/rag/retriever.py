from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
load_dotenv()

texts = ["Paris is the capital of France", "Tokyo is in Japan", "Soccer is popular in Europe"]
embedding_model = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts, embedding_model)

# Create retriever
retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 1})
# Query
docs = retriever.invoke("What is the capital of France?")
print(docs[0].page_content)  # → "Paris is the capital of France"
