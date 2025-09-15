import os
import json
import uuid
import fitz  # PyMuPDF
from langchain.docstore.document import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from load_vector_dbs.load_dbs import load_vector_database
from data_preparation.image_data_prep import ImageDescription


def process_pdf_and_stream(uploaded_pdf_path: str):
    if not os.path.exists(uploaded_pdf_path):
        yield f"Error: File does not exist: {uploaded_pdf_path}"
        return

    try:
        yield f"Processing document: {uploaded_pdf_path}"
        pdf_document = fitz.open(uploaded_pdf_path)
        source_file_name = os.path.basename(uploaded_pdf_path)

        embeddings = OpenAIEmbeddings()
        db_init = load_vector_database()

        # --- Text ingestion ---
        retriever, text_vectorstore, _ = db_init.get_text_retriever()
        existing_files = db_init.get_vector_store_files(text_vectorstore)
        print(f"Existing ingested files: {existing_files}")    
        if source_file_name in existing_files:
            yield f"{source_file_name} already ingested (text). Skipping text ingestion."
        else:
            documents = []
            for page_num, page in enumerate(pdf_document):
                text = page.get_text("text")
                if text.strip():
                    metadata = {"source_file": source_file_name,
                                "page_num": page_num + 1}
                    documents.append(Document(page_content=text, metadata=metadata))

            if documents:
                yield "Extracted text and tables from PDF."
                text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
                    chunk_size=1000, chunk_overlap=100
                )
                text_chunks = text_splitter.split_documents(documents)

                # Generate deterministic UUIDs
                ids = [
                    str(uuid.uuid5(uuid.NAMESPACE_DNS,
                                   f"{doc.metadata['source_file']}_page{doc.metadata['page_num']}_{i}"))
                    for i, doc in enumerate(text_chunks)
                ]

                text_vectorstore.add_documents(text_chunks, ids=ids)
                yield f"Added text chunks from {source_file_name} into Qdrant text vector store."
            else:
                yield "No text extracted from PDF."

        # --- Image ingestion ---
        image_vectorstore, _, _ = db_init.get_image_retriever()
        existing_imgs = db_init.get_img_vector_store_companies(image_vectorstore)

        if source_file_name in existing_imgs:
            yield f"{source_file_name} already ingested (images). Skipping image ingestion."
        else:
            yield f"Extracting images from {source_file_name}..."
            img_processor = ImageDescription(uploaded_pdf_path)
            image_info = img_processor.get_image_information()

            if image_info:
                metadata_path = f"metadata_{source_file_name}.json"
                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(image_info, f, indent=2)
                yield f"Saved image metadata to {metadata_path}"

                image_documents = img_processor.getRetriever(
                    metadata_path, os.path.splitext(source_file_name)[0])

                # Generate deterministic UUIDs for images
                img_ids = [
                    str(uuid.uuid5(uuid.NAMESPACE_DNS,
                                   f"{doc.metadata.get('company','NA')}_{source_file_name}_{i}"))
                    for i, doc in enumerate(image_documents)
                ]

                image_vectorstore.add_documents(image_documents, ids=img_ids)
                yield f"Added image captions from {source_file_name} into Qdrant image vector store."
            else:
                yield "No images found in PDF."

        yield f"Completed ingestion for {source_file_name}"

    except Exception as e:
        yield f"Error while processing PDF {uploaded_pdf_path}: {str(e)}"
