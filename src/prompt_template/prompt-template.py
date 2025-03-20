from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
load_dotenv()

llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.7,
    max_tokens=1000,
    verbose=True
)

# # prompt template
# prompt = ChatPromptTemplate.from_template("Tell me a joke about {subject}")
#
# # Create a LLM chain
# chain = prompt | llm # output of prompt is the input of llm
#
# llm_response = chain.invoke({"subject": "dog"})
#
# print(llm_response)


# Another way of creating prompt

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Generate a list of 10 synonyms for the following word. Return the result as a comma separated."),
        ("human", "{input}")
    ]
)

chain = prompt | llm

llm_response = chain.invoke({"input": "happy"})
print(llm_response)