# https://python.langchain.com/v0.1/docs/modules/data_connection/vectorstores/
# https://python.langchain.com/docs/integrations/vectorstores/
# Source ---> Load ---> Transform ---> Embed ---> Store ---> Retrieve
from typing import Dict
from dotenv import load_dotenv
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores.faiss import FAISS  # In-memory vector store
from langchain_core.runnables import RunnablePassthrough

load_dotenv()


def get_documents_from_web(url):
    loader = WebBaseLoader(
        url)  # scrape the data from the web page. see here https://python.langchain.com/docs/concepts/document_loaders/
    raw_documents = loader.load()  # add the contents of that page to the langchain document.
    print("raw documents", raw_documents)

    # If we directly used the document scraped above, the token size would be too big, and we want to feed only part of it using some transform.
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    split_docs = splitter.split_documents(raw_documents)
    print("split docs:")
    for doc in split_docs:
        print(doc)
    return split_docs


def create_vector(documents):
    """ Convert the documents into the format that the database(vector database) will understand. Embedding function."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store


def parse_retriever_input(params: Dict):
    print("params", params)
    return params["question"].content


def create_chain(vector_store):
    model = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.4,
        max_tokens=1000,
    )

    prompt = ChatPromptTemplate.from_template("""
    Answer the user's question:
    Context: {context}
    Question: {question}
    """)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 2})  # by default, it is top 5 documents. we can increase or decrease by kwargs.

    document_chain = create_stuff_documents_chain(
        llm=model,
        prompt=prompt
    )

    '''
    1. Start with passthrough → input (question) is preserved.
    2. .assign(context=...)
        Extract the question text (parse_retriever_input).
        Pass it to the retriever.
        Output (retrieved docs) → assigned to context.
    3. .assign(answer=document_chain)
        Take {question, context} → feed into document_chain.
        Output from LLM → assigned to answer.
        '''
    retrieval_chain = (RunnablePassthrough
                       .assign(context=parse_retriever_input | retriever, )
                       .assign(answer=document_chain, )
                       )

    return retrieval_chain


split_docs = get_documents_from_web("https://python.langchain.com/v0.1/docs/expression_language/")
vector_store = create_vector(split_docs)
chain = create_chain(vector_store)

response = chain.invoke({
    "question": HumanMessage("What is LCEL")}
)

'''
Final answer is like this

{
  'question': HumanMessage(content='What is LCEL', additional_kwargs={}, response_metadata={}),
   'context': [
         Document(id='afd19d60-378a-4499-8e9f-051a1eb70e3b', metadata={'source': 'https://python.langchain.com/v0.1/docs/expression_language/', 'title': 'LangChain Expression Language (LCEL) | 🦜️🔗 LangChain', 'description': 'LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together.', 'language': 'en'}, page_content='Expression Language, or LCEL, is a declarative way to easily compose chains together.'),
         Document(id='89d9f3a8-b8f1-4581-a8fd-b329b9391505', metadata={'source': 'https://python.langa≈Ωachain.com/v0.1/docs/expression_language/', 'title': 'LangChain Expression Language (LCEL) | 🦜️🔗 LangChain', 'description': 'LangChain Expression Language, or LCEL, is a declarative way to easily compose chains together.', 'language': 'en'}, page_content='LangChain Expression Language (LCEL) | 🦜️🔗 LangChain')
    ],
   'answer': 'LCEL stands for LangChain Expression Language. It is a declarative way to easily compose chains together in the LangChain platform. It allows users to efficiently combine different elements and create complex chains for various purposes.'
}
'''
print("response", response)
print("context", response["context"])
print("Answer", response["answer"])


