from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()



# no need to provide api_key as it automatically look for OPENAI_API_KEY env variable
# otherwise we have to call it like llm = ChatOpenAI(api_key="api key")
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7, # 0 is strict and factual (eg: for answering from DB or math) and 1 is more creative.
    max_tokens=1000, # limit the token
    verbose=True
)
# read here for more parameters https://python.langchain.com/docs/concepts/chat_models/

llm_response = llm.invoke("Hello, How are you?") # for multiple prompts use batch see example below.
# llm_response = llm.batch(["Hello, how are you?", "Write a poem about AI"]) #we will get multiple answer. Check by uncommenting it.

# For chunk response use stream method instead of invoke see example below:
response = llm.stream("Write a poem about AI?")
# for chunk in response:
#     print(chunk.content, end="", flush=True)

print(llm_response)