# It has to be fixed
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import MessagesPlaceholder
from langchain.chains.history_aware_retriever import create_history_aware_retriever

load_dotenv()

def get_documents_from_web(url):
    loader = WebBaseLoader(url)
    raw_documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400, # increase the response quality with higher value
        chunk_overlap=20
    )
    split_docs = splitter.split_documents(raw_documents)
    return split_docs


def create_vector(documents):
    """ Convert the documents into the format that the database(vector database) will understand. Embedding function."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def create_chain(vector_store):
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.4
    )

    # prompt = ChatPromptTemplate.from_template("""
    # Answer the user's question:
    # Context: {context}
    # Question: {question}
    # """)

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user's questions based on the context: {context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])


    retriever = vector_store.as_retriever(search_kwargs={"k":3})
    retriever_prompt = ChatPromptTemplate.from_messages([
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
        ("human", "given the above conversation, generate a search query to look up in order to get information relevant to the conversation")
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm = model,
        retriever=retriever,
        prompt=retriever_prompt
    )
    # Define the retrieval chain
    retrieval_chain = (
            RunnableParallel({"context": history_aware_retriever, "question": RunnablePassthrough()})
            | prompt
            | model
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

    chat_history = []
    while True:
        user_input = input("You: ")
        if user_input.lower() == "exit":
            break

        response = process_chat(chain, user_input, chat_history)
        chat_history.append(HumanMessage(content=user_input))
        chat_history.append(AIMessage(content=response))

        print("Assistant: ", response)

