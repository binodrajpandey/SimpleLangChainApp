from langchain_core.runnables import RunnablePassthrough

chain = RunnablePassthrough()
# returns the input as-is.
result = chain.invoke({"question": "What is RAG?"})
print(type(result))
print(result)

# assign new key-value
#input (question) is passed through, AND a new field (context) was added.
chain = RunnablePassthrough.assign(
    context=lambda x: "retrieved docs for: " + x["question"]
)
result = chain.invoke({"question": "What is RAG?"})
print(result) # {'question': 'What is RAG?', 'context': 'retrieved docs for: What is RAG?'}
