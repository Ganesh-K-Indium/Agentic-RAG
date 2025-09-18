import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()
# ---------------------------
# Jira credentials (from .env or hardcoded for testing)
# ---------------------------
JIRA_URL = os.getenv('JIRA_URL')
EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "<PUT_YOUR_JIRA_API_TOKEN>")
OUTPUT_DIR = "jira_attachments"

# ---------------------------
# Download attachments from a Jira issue
# ---------------------------
def download_attachments(issue_key: str):
    url = f"{JIRA_URL}/rest/api/3/issue/{issue_key}?fields=attachment"
    auth = HTTPBasicAuth(EMAIL, JIRA_API_TOKEN)
    headers = {"Accept": "application/json"}

    response = requests.get(url, headers=headers, auth=auth)

    if response.status_code != 200:
        print(f" Failed to fetch issue {issue_key}: {response.status_code}, {response.text}")
        return []

    issue_data = response.json()
    attachments = issue_data.get("fields", {}).get("attachment", [])

    if not attachments:
        print(f"No attachments found in issue {issue_key}.")
        return []

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    downloaded_files = []

    for attachment in attachments:
        file_name = attachment["filename"]
        file_url = attachment["content"]

        print(f"📥 Downloading: {file_name}")
        file_response = requests.get(file_url, headers=headers, auth=auth, stream=True)

        if file_response.status_code == 200:
            local_path = os.path.join(OUTPUT_DIR, file_name)
            with open(local_path, "wb") as f:
                for chunk in file_response.iter_content(chunk_size=8192):
                    f.write(chunk)
            downloaded_files.append(local_path)
            print(f"✅ Saved: {local_path}")
        else:
            print(f"⚠️ Failed to download {file_name}: {file_response.status_code}")

    return downloaded_files


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    ISSUE_KEY = "TEST-1"  # 🔹 replace with your Jira ticket key
    files = download_attachments(ISSUE_KEY)
    print(f"📊 Downloaded {len(files)} files from {ISSUE_KEY}.")
