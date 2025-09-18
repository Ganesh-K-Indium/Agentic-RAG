# dropbox_download.py
import os
import io
import socket
import dropbox
from dropbox.exceptions import AuthError, ApiError
from googleapiclient.errors import HttpError


def check_host_reachable(host="api.dropboxapi.com", port=443, timeout=5) -> bool:
    """Check if Dropbox API host is reachable (network test)."""
    try:
        socket.setdefaulttimeout(timeout)
        sock = socket.create_connection((host, port))
        sock.close()
        return True
    except OSError:
        return False


def download_pdfs_from_dropbox(folder_path: str, local_folder: str, access_token: str) -> list:
    """
    Download all PDFs from a Dropbox folder.

    Args:
        folder_path: Dropbox folder path (e.g., "/MyFolder")
        local_folder: Local folder to save PDFs
        access_token: Dropbox API access token

    Returns:
        List of local file paths for downloaded PDFs
    """
    if not check_host_reachable():
        print("Network issue: Cannot reach api.dropboxapi.com. Check firewall/proxy/DNS.")
        return []

    try:
        dbx = dropbox.Dropbox(access_token)
        dbx.users_get_current_account()  # Token validation
    except AuthError:
        print("Invalid Dropbox access token. Please check your .env or token string.")
        return []
    except Exception as e:
        print(f"Unexpected error during auth: {e}")
        return []

    os.makedirs(local_folder, exist_ok=True)
    downloaded_files = []

    try:
        result = dbx.files_list_folder(folder_path)

        for entry in result.entries:
            if isinstance(entry, dropbox.files.FileMetadata) and entry.name.endswith(".pdf"):
                local_path = os.path.join(local_folder, entry.name)
                print(f"Downloading: {entry.name}")

                _, res = dbx.files_download(entry.path_lower)
                with open(local_path, "wb") as f:
                    f.write(res.content)

                downloaded_files.append(local_path)
                print(f"Saved to {local_path}")

    except ApiError as e:
        print(f"Dropbox API error: {e}")
    except Exception as e:
        print(f"General error: {e}")

    print(f"📂 Downloaded {len(downloaded_files)} PDFs.")
    return downloaded_files


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    ACCESS_TOKEN = os.getenv("DROPBOX_ACCESS_TOKEN")
    FOLDER_PATH = "/Agentic-AI"   # Change to your folder in Dropbox
    LOCAL_FOLDER = "downloaded_dropbox_pdfs"

    if not ACCESS_TOKEN:
        print("Missing DROPBOX_ACCESS_TOKEN in .env file")
    else:
        download_pdfs_from_dropbox(FOLDER_PATH, LOCAL_FOLDER, ACCESS_TOKEN)
