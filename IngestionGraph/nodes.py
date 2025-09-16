import os
from fastapi.responses import StreamingResponse
from fastapi import UploadFile, File
from .graph_state import IngestionState
from IngestionGraph.utils.pdf_processor import process_pdf_and_stream  # assuming you moved process_pdf_and_stream to utils
from IngestionGraph.utils.confluence import download_all_pdfs  # your confluence downloader
from IngestionGraph.utils.gdrive import download_pdfs_from_folder
from typing import Generator

# Local PDF ingestion node
def ingest_local_pdf(state: IngestionState):
    file_name = state.get("file_name")
    logs = state["logs"]

    if not file_name:
        msg = "No file_name provided for local PDF ingestion."
        logs.append(msg)
        print(msg)
        return state

    file_path = os.path.join("10k_PDFs", file_name)
    if not os.path.exists(file_path):
        msg = f"File not found: {file_path}"
        logs.append(msg)
        print(msg)
        return state

    msg = f"Processing local PDF: {file_name}"
    logs.append(msg)
    print(msg)

    for update in process_pdf_and_stream(file_path):
        logs.append(update)
        print(update)

    state["logs"] = logs
    print("Ingestion completed successfully.")
    return state


# Confluence ingestion node
def ingest_confluence(state: IngestionState):
    logs = state.get("logs", [])
    space_key = state.get("space_key")

    if not space_key:
        msg = "No space_key provided for Confluence ingestion."
        print(msg)
        logs.append(msg)
        state["status"] = "error"
        state["logs"] = logs
        return state

    msg = f"Downloading PDFs from Confluence space {space_key}..."
    print(msg)
    logs.append(msg)

    pdf_files = download_all_pdfs()

    if not pdf_files:
        msg = "No PDFs found in the specified Confluence space."
        print(msg)
        logs.append(msg)
        state["status"] = "no_files"
        state["logs"] = logs
        return state

    for pdf_path in pdf_files:
        msg = f" Downloaded: {pdf_path}"
        print(msg)
        logs.append(msg)

        for update in process_pdf_and_stream(pdf_path):
            print(update)      #  stream to console
            logs.append(update)

    msg = "Completed Confluence ingestion."
    print(msg)
    logs.append(msg)

    state["status"] = "success"
    state["logs"] = logs
    return state


# Jira stub
def ingest_jira(state: IngestionState):
    ticket_id = state.get("ticket_id")
    logs = state["logs"]
    logs.append(f"[LOG] Pretending to ingest Jira ticket {ticket_id}")
    state["logs"] = logs
    return state


# SharePoint stub
def ingest_sharepoint(state: IngestionState):
    file_url = state.get("file_url")
    logs = state["logs"]
    logs.append(f"[LOG] Pretending to ingest SharePoint file from {file_url}")
    state["logs"] = logs
    return state




def ingest_gdrive_folder(state: IngestionState):
    logs = state.get("logs", [])
    folder_id = state.get("folder_id")

    if not folder_id:
        msg = "No folder_id provided for Google Drive ingestion."
        print(msg)
        logs.append(msg)
        state["status"] = "error"
        state["logs"] = logs
        return state

    msg = f"Downloading PDFs from Google Drive folder {folder_id}..."
    print(msg)
    logs.append(msg)

    pdf_files = download_pdfs_from_folder(folder_id)

    if not pdf_files:
        msg = "No PDFs found in the specified Google Drive folder."
        print(msg)
        logs.append(msg)
        state["status"] = "no_files"
        state["logs"] = logs
        return state

    for pdf_path in pdf_files:
        msg = f" Downloaded: {pdf_path}"
        print(msg)
        logs.append(msg)

        for update in process_pdf_and_stream(pdf_path):
            print(update)
            logs.append(update)

    msg = "Completed Google Drive ingestion."
    print(msg)
    logs.append(msg)

    state["status"] = "success"
    state["logs"] = logs
    return state
