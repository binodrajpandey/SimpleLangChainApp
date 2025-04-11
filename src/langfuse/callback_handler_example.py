import os
import uuid

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langfuse.callback import CallbackHandler
from langfuse import Langfuse

load_dotenv()

langfuse = Langfuse()
trace_client = langfuse.trace(
            name="first trace",
            session_id=str(uuid.uuid4()),
            user_id="user-1",
            tags=[f"client_binod"]
        )

# This integration captures inputs, outputs, and model usage automatically.
callback_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    stateful_client=trace_client # can be created without it.
)

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=1000,
    verbose=True,
    # callbacks=[callback_handler]
)

response = llm.invoke(input="What is langfuse?", config={"callbacks": [callback_handler]})
print(response.content)
