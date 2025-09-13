from .graph_state import IngestionState

def route_ingestion(state: IngestionState) -> str:
    """
    Decide which ingestion node to route to based on `state.source`.
    """
    source = state.get("source", "").lower()

    if source == "local_pdf":
        return "local_pdf"
    elif source == "confluence":
        return "confluence"
    elif source == "jira":
        return "jira"
    elif source == "sharepoint":
        return "sharepoint"
    else:
        return "error"  # a fallback node

def add_edges(workflow):
    # Conditional routing from 'start' node
    workflow.add_conditional_edges(
        "start",
        route_ingestion,
        {
            "local_pdf": "local_pdf",
            "confluence": "confluence",
            "jira": "jira",
            "sharepoint": "sharepoint",
            "error": "error",
        },
    )
