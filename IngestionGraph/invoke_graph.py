# IngestionGraph/invoke_graph.py
from langgraph.graph import StateGraph, END
from .graph_state import IngestionState
from .nodes import ingest_local_pdf, ingest_confluence, ingest_jira, ingest_sharepoint, ingest_gdrive_folder
from .edges import route_ingestion


class IngestionGraph:
    def __init__(self):
        self.workflow = StateGraph(IngestionState)

        self.workflow.add_node("start", self._start_node)
        self.workflow.set_entry_point("start")

        self.workflow.add_node("local_pdf", ingest_local_pdf)
        self.workflow.add_node("confluence", ingest_confluence)
        self.workflow.add_node("jira", ingest_jira)
        self.workflow.add_node("sharepoint", ingest_sharepoint)
        self.workflow.add_node("gdrive_folder", ingest_gdrive_folder)

        self.workflow.add_node("error", self._error_node)

        self.workflow.add_conditional_edges(
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

        self.workflow.add_edge("local_pdf", END)
        self.workflow.add_edge("confluence", END)
        self.workflow.add_edge("jira", END)
        self.workflow.add_edge("sharepoint", END)
        self.workflow.add_edge("gdrive_folder", END)
        self.workflow.add_edge("error", END)

        self.app = self.workflow.compile()

    # -------------------------
    # helper nodes
    # -------------------------
    def _start_node(self, state: IngestionState):
        """
        Entrypoint node: ensure required keys exist and return state.
        """
        # ensure simple defaults so downstream nodes don't crash
        if "logs" not in state or state["logs"] is None:
            state["logs"] = []
        if "source" not in state:
            # leave it blank — route_ingestion will handle unknowns
            state["source"] = ""
        return state

    def _error_node(self, state: IngestionState):
        logs = state.get("logs", [])
        logs.append("Unknown ingestion source; no action taken.")
        state["logs"] = logs
        state["status"] = "error"
        return state

    def get_graph(self):
        return self.app
