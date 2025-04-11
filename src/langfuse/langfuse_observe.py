import os

from dotenv import load_dotenv
from langfuse import Langfuse
from langfuse.decorators import observe

load_dotenv()
# langfuse = Langfuse(
#     public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
#     secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
#     host=os.getenv("LANGFUSE_HOST")
# )

@observe(capture_output=False)  # This will automatically create a trace
def fetch_data(input):
    print("Input data...", input)
    return {"data": f"Sample data {input}"}

fetch_data("Hey Binod")
