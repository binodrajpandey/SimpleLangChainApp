from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
load_dotenv()
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=1000
)

def get_llm():
    return llm
