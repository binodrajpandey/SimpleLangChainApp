from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain


load_dotenv()

docA = Document(
    page_content="LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains (we’ve seen folks successfully run LCEL chains with 100s of steps in production)."
)

model = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7
)

prompt = ChatPromptTemplate.from_template("""
Answer the user's question:
Context: {context}
Question: {input}
""")

# OR we could do directly as below
# prompt = ChatPromptTemplate.from_template(f"""
# Answer the user's question:
# Context: LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together. LCEL was designed from day 1 to support putting prototypes in production, with no code changes, from the simplest “prompt + LLM” chain to the most complex chains (we’ve seen folks successfully run LCEL chains with 100s of steps in production).
# Question: {input}
# """)

# We can create a chain in either way
# chain = prompt | model
chain = create_stuff_documents_chain(
    llm=model,
    prompt=prompt
)

response = chain.invoke({
    "input": "What is LCEL",
    "context": [docA]
}) # might return wrong content without context.

print(response)

# Fetch data from the source using document loader.
# The document loader will store the content of this data source what is known as langchain document.
