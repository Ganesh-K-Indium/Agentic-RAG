from typing import TypedDict, List, Optional

class IngestionState(TypedDict):
    request: str
    logs: List[str]
    source: str                     # local_pdf, confluence, jira, sharepoint, gdrive_folder
    file_name: Optional[str]        # For local PDF
    space_key: Optional[str]        # For Confluence
    project_key: Optional[str]      # For Jira
    file_url: Optional[str]         # For SharePoint
    folder_id: Optional[str]        # For Google Drive folder
