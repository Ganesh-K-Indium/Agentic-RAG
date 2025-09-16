from .graph_state import IngestionState

def route_ingestion(state: IngestionState) -> str:
    source = state.get("source", "").lower()

    if source == "local_pdf":
        return "local_pdf"
    elif source == "confluence":
        return "confluence"
    elif source == "jira":
        return "jira"
    elif source == "sharepoint":
        return "sharepoint"
    elif source == "gdrive_folder":
        return "gdrive_folder"
    else:
        return "error"

def add_edges(workflow):
    workflow.add_conditional_edges(
        "start",
        route_ingestion,
        {
            "local_pdf": "local_pdf",
            "confluence": "confluence",
            "jira": "jira",
            "sharepoint": "sharepoint",
            "gdrive_folder": "gdrive_folder",
            "error": "error",
        },
    )

