from langchain.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma



# LOAD OUR FILE 
loader = CSVLoader( file_path="..\Data\AbsherQADocs.csv")
data = loader.load()


# CREATE EMBEDDING FUNCTION
STembadding = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")


# CREATE VECTOR STORE
vector_db = Chroma.from_documents(documents=data,embedding=STembadding , persist_directory="../VectorDB")

vector_db.persist()
