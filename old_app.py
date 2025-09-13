import os
import fitz
from pydantic import BaseModel
from langchain.schema import Document
from langchain_openai import OpenAIEmbeddings
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import StreamingResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from load_vector_dbs.load_dbs import load_vector_database
from data_preparation.image_data_prep import ImageDescription
from Graph.invoke_graph import BuildingGraph

app = FastAPI()
Agent = BuildingGraph().get_graph()

class QueryInput(BaseModel):
    query: str

def process_pdf_and_stream(uploaded_pdf_path):
    """Generator function to process PDF and stream intermediate updates."""
    if not os.path.exists(uploaded_pdf_path):
        yield "Error: File does not exist.\n"
        return

    yield "Processing document\n"

    new_documents = []
    pdf_document = fitz.open(uploaded_pdf_path)
    source_file_name = os.path.basename(uploaded_pdf_path)
     
    for page_num, _ in enumerate(pdf_document):
        page = pdf_document[page_num]
        text = page.get_text("text")
        metadata = {"source_file": source_file_name, "page_num": page_num + 1}
        new_documents.append(Document(page_content=text, metadata=metadata))
    
    yield "Extracted text and tables from PDFs.\n"

    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=1000, chunk_overlap=100)
    new_doc_splits = text_splitter.split_documents(new_documents)
    embeddings = OpenAIEmbeddings()
    init = load_vector_database()
    _, text_vectorstore, text_vector_db_path = init.get_text_retriever()

    text_vectorstore.add_documents(new_doc_splits, embedding=embeddings)
    text_vectorstore.save_local(text_vector_db_path)

    yield f"Successfully added Text and tables from {source_file_name} to FAISS text_table vector store!\n"
    
    yield f"Working on adding description of images of {source_file_name} to FAISS image vector store!\n"
    
    init_img = ImageDescription(uploaded_pdf_path)
    image_info = init_img.get_image_information()
    image_description_file_path = init_img.get_image_description(image_info)
    
    company_name = image_description_file_path.split("\\")[-1].replace(".json","")
    image_documents = init_img.getRetriever(image_description_file_path, company_name)
    image_vectorstore_10k, _, image_vector_db_path = init.get_image_retriever()
    yield "Got the description of images. adding document into image vector store!\n"
    
    image_vectorstore_10k.add_documents(image_documents, embedding=embeddings)
    image_vectorstore_10k.save_local(image_vector_db_path)
    yield "Added Images into image vector store!\n"
    
@app.post("/upload-pdf/")
async def upload_pdf(file: UploadFile = File(...)):
    """API endpoint to upload a PDF and stream processing updates."""
    file_path = os.path.join(r"10k_PDFs", file.filename)
    os.makedirs(r"10k_PDFs", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(await file.read())

    return StreamingResponse(process_pdf_and_stream(file_path), media_type="text/plain")

@app.post("/ask")
async def ask_agent(payload: QueryInput):
    question = payload.query
    inputs = {
    "messages": [question]
    }

    result = Agent.invoke(inputs) 
    return {"answer": result}
