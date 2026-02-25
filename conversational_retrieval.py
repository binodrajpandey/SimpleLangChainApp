# reference: https://python.langchain.com/v0.1/docs/use_cases/chatbots/retrieval/
from typing import Dict

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
load_dotenv()

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    raw_documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=20
    )
    return splitter.split_documents(raw_documents)


def create_vector(documents):
    """ Convert the documents into the format that the database(vector database) will understand. Embedding function."""
    embeddings = OpenAIEmbeddings()
    return FAISS.from_documents(documents, embeddings)

def parse_retriever_input(params: Dict):
    return params["question"]

def create_chain(vector_store):
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.4
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    retriever = vector_store.as_retriever(search_kwargs={"k":3})
    document_chain = prompt | model
    retrieval_chain = RunnablePassthrough.assign(
        context=parse_retriever_input | retriever,
    ).assign(
        answer=document_chain,
    )

    return retrieval_chain

def process_chat(chain, question, chat_history):
    response = chain.invoke({
        "question": question,
        "chat_history": chat_history
    })
    return response["answer"]


if __name__ == "__main__":
    split_docs = get_documents_from_web("https://python.langchain.com/v0.1/docs/expression_language/")
    vector_store = create_vector(split_docs)
    chain = create_chain(vector_store)

    # chat_history = [
    #     HumanMessage(content="Hello"),
    #     AIMessage(content="Hello, How can I assist you?"),
    #     HumanMessage(content="My name is Binod")
    # ]

    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print("Assistant: ", response)

