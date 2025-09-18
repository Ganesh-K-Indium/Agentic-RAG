import os
import requests
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv

load_dotenv()

# ---------------------------
# Jira credentials (from .env)
# ---------------------------
JIRA_URL = os.getenv('JIRA_URL')  # e.g., https://yourdomain.atlassian.net
EMAIL = os.getenv("JIRA_EMAIL")
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN")
OUTPUT_DIR = "jira_attachments"

if not JIRA_URL or not EMAIL or not JIRA_API_TOKEN:
    raise ValueError("Missing JIRA_URL, JIRA_EMAIL, or JIRA_API_TOKEN in .env")


def get_issues(project_key: str):
    """Fetch all issues from a Jira project using JQL."""
    jql = f"project={project_key}"  # you can customize this
    url = f"{JIRA_URL}/rest/api/3/search"
    auth = HTTPBasicAuth(EMAIL, JIRA_API_TOKEN)
    headers = {"Accept": "application/json"}

    issues = []
    start_at = 0
    max_results = 50  # pagination

    while True:
        params = {"jql": jql, "fields": "attachment", "startAt": start_at, "maxResults": max_results}
        response = requests.get(url, headers=headers, auth=auth, params=params)

        if response.status_code != 200:
            print(f"Failed to fetch issues: {response.status_code}, {response.text}")
            break

        data = response.json()
        issues.extend(data.get("issues", []))

        if start_at + max_results >= data.get("total", 0):
            break

        start_at += max_results

    return issues


def download_attachments_from_issue(issue):
    """Download all attachments from a given issue JSON object."""
    issue_key = issue["key"]
    attachments = issue.get("fields", {}).get("attachment", [])

    if not attachments:
        print(f"No attachments in {issue_key}")
        return []

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    downloaded_files = []

    for attachment in attachments:
        file_name = attachment["filename"]
        file_url = attachment["content"]

        print(f"📥 Downloading from {issue_key}: {file_name}")
        response = requests.get(file_url, auth=HTTPBasicAuth(EMAIL, JIRA_API_TOKEN), stream=True)

        if response.status_code == 200:
            local_path = os.path.join(OUTPUT_DIR, f"{issue_key}_{file_name}")
            with open(local_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            downloaded_files.append(local_path)
            print(f"Saved: {local_path}")
        else:
            print(f"Failed to download {file_name}: {response.status_code}")

    return downloaded_files


if __name__ == "__main__":
    PROJECT_KEY = "TEST"  # change if your project key is different
    issues = get_issues(PROJECT_KEY)

    print(f"Found {len(issues)} issues in {PROJECT_KEY}")

    total_files = 0
    for issue in issues:
        files = download_attachments_from_issue(issue)
        total_files += len(files)

    print(f"Downloaded {total_files} attachments from project {PROJECT_KEY}")
