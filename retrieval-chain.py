# https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/
# https://python.langchain.com/docs/integrations/vectorstores/
# Source ---> Load ---> Transform ---> Embed ---> Store ---> Retrieve
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS # In-memory vector store
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

def get_documents_from_web(url):
    loader = WebBaseLoader(url) # scrape the data from the web page.
    raw_documents = loader.load() # add the contents of that page to the langchain document.
    print(raw_documents)

    # If we directly used the document scraped above, the token size would be big and we want to feed only part of it using some transform.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    split_docs = splitter.split_documents(raw_documents)
    print(split_docs)

    return split_docs


def create_vector(documents):
    """ Convert the documents into the format that the database(vector database) will understand. Embedding function."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    print(vector_store)
    return vector_store

def create_chain(vector_store):
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.4
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question:
    Context: {context}
    Question: {question}
    """)


    retriever = vector_store.as_retriever(search_kwargs={"k":2}) # by default, it is top 5 documents. we can increase or decrease by kwargs.

    # Define the retrieval chain
    retrieval_chain = (
            RunnableParallel({"context": retriever, "question": RunnablePassthrough()})
            | prompt
            | model
    )

    return retrieval_chain


split_docs = get_documents_from_web("https://python.langchain.com/v0.1/docs/expression_language/")
vector_store = create_vector(split_docs)

response = create_chain(vector_store).invoke({
    "question": "What is LCEL",
})

print(response["context"])
print(response)
