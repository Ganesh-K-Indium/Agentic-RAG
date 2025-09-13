from typing import TypedDict, List, Optional

class IngestionState(TypedDict):
    request: str
    logs: List[str]
    source: str                   # local_pdf, confluence, jira, sharepoint
    file_name: Optional[str]      # For local PDF
    space_key: Optional[str]      # For Confluence
    ticket_id: Optional[str]      # For Jira
    file_url: Optional[str]       # For SharePoint
