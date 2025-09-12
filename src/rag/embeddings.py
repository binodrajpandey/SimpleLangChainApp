from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
load_dotenv()
text1 = "The cat sat on the mat"
text2 = "A feline rested on the carpet"
text3 = "Python is a programming language"


# Initialize the embeddings model
embeddings_model = OpenAIEmbeddings()

embeddings = embeddings_model.embed_documents([text1, text2, text3])

print("number of documents:", len(embeddings))
print(embeddings[0])
print("Dimension:", len(embeddings[0]))
print(embeddings[1])
print("Dimension:", len(embeddings[1]))
print(embeddings[1])
print("Dimension:", len(embeddings[1]))
