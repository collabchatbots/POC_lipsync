import os

from langchain.embeddings.openai import OpenAIEmbeddings
#from langchain.vectorstores import Chroma
from langchain.text_splitter import TokenTextSplitter
from langchain.document_loaders import DirectoryLoader
from langchain.vectorstores import FAISS

os.environ["OPENAI_API_KEY"] = "TestTestsk-nEzxLOHuHX67HfDqW72kT3BlbkFJeWenIo3mxx5ifk6peuCS"#"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"

def get_documents():
    loader = DirectoryLoader('bot/demo_files/', glob="**/*.txt")
    data = loader.load()
    #print(len(data))
    return data

def create_db():
    persist_directory = "demo_db"

    if not os.path.exists(persist_directory):
        db_data = get_documents()

        text_splitter = TokenTextSplitter(chunk_size=1800, chunk_overlap=0)
        db_doc = text_splitter.split_documents(db_data)
        
        embeddings = OpenAIEmbeddings()
        vectordb = FAISS.from_documents(db_doc, embeddings)
        vectordb.save_local(persist_directory)
        #print("Vector database created.")
    #else:
        #print("Vector database already exists. Skipping creation.")
