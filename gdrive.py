# download_gdrive_folder.py
import os
import io
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account


def download_gdrive_folder(folder_id: str, local_folder: str, credentials_path: str) -> list:
    """
    Download all PDFs from a Google Drive folder using a service account.

    Args:
        folder_id: Google Drive folder ID (from URL)
        local_folder: Local folder to save PDFs
        credentials_path: Path to service account JSON credentials

    Returns:
        List of local file paths for downloaded PDFs
    """
    creds = service_account.Credentials.from_service_account_file(
        credentials_path, scopes=["https://www.googleapis.com/auth/drive.readonly"]
    )
    service = build("drive", "v3", credentials=creds)

    os.makedirs(local_folder, exist_ok=True)

    # Query all PDF files in the folder
    query = f"'{folder_id}' in parents and mimeType='application/pdf'"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    files = results.get("files", [])

    if not files:
        print("No PDFs found in folder.")
        return []

    downloaded_files = []

    for file in files:
        file_id = file["id"]
        file_name = file["name"]
        local_path = os.path.join(local_folder, file_name)

        print(f"Downloading: {file_name}")

        request = service.files().get_media(fileId=file_id)
        with io.FileIO(local_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                status, done = downloader.next_chunk()
                if status:
                    print(f"Progress {int(status.progress() * 100)}%")

        downloaded_files.append(local_path)
        print(f"Saved to {local_path}")

    return downloaded_files


if __name__ == "__main__":
    FOLDER_ID = "15uqZb5Ubzy7XV4iSw9EPlunnvXOGZnRn"  # from Drive folder URL
    LOCAL_FOLDER = "downloaded_pdfs"
    CREDS_PATH = "service_account.json"

    try:
        files = download_gdrive_folder(FOLDER_ID, LOCAL_FOLDER, CREDS_PATH)
        print(f"Downloaded {len(files)} PDFs.")
    except Exception as e:
        print(f"Error: {e}")
