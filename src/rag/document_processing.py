from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

url = "https://python.langchain.com/v0.1/docs/expression_language/"
loader = WebBaseLoader(url)  # scrape the data from the web page.
raw_documents = loader.load()  # add the contents of that page to the langchain document.
print(raw_documents)

# If we directly used the document scraped above, the token size would be too big, and we want to feed only part of it using some transform.
splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
split_docs = splitter.split_documents(raw_documents)
for split_doc in split_docs:
    print(split_doc)
