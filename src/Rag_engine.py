import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma 
from src.config import settings

def initialize_rag_system(pdf_path: str):

    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"the specified PDF file does not exist: {pdf_path}")
        
    print("Loading the PDF document and splitting it into chunks...")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE, 
        chunk_overlap=settings.CHUNK_OVERLAP
    )

    chunks = text_splitter.split_documents(documents)
    
    print(f"The file has been split into {len(chunks)} chunks. Generating embeddings...")
    

    embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)
    
    print("The embeddings are being generated and saved to ChromaDB...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=settings.DB_DIR
    )
    print("The embeddings are saved to ChromaDB successfully.")
    return vector_store

def get_retriever():
    embeddings = GoogleGenerativeAIEmbeddings(model=settings.EMBEDDING_MODEL)
    vector_store = Chroma(persist_directory=settings.DB_DIR, embedding_function=embeddings)
    return vector_store.as_retriever(search_kwargs={"k": 3}) # Top 3