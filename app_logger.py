# file: logger.py
import os
import json
import datetime

def format_graph_output(data: dict) -> str:
    """Format RAG graph output into comprehensive Markdown with all details."""
    lines = []
    
    # Handle new comprehensive response format
    if "answer" in data and isinstance(data["answer"], str):
        # Main answer section
        lines.append("## Answer")
        lines.append(data["answer"])
        lines.append("")
        
        # Document Citations Section
        if data.get("citations"):
            lines.append("## Document Citations")
            for idx, citation in enumerate(data["citations"], 1):
                lines.append(f"### Citation {idx}")
                lines.append(f"- **Source Type**: {citation.get('source_type', 'Unknown')}")
                lines.append(f"- **Document ID**: {citation.get('document_id', 'N/A')}")
                lines.append(f"- **Relevance Score**: {citation.get('relevance_score', 'N/A')}")
                lines.append(f"- **Key Information**: {citation.get('key_information', 'N/A')}")
                lines.append("")
        
        # Document Details Section  
        if data.get("documents"):
            lines.append("## Document Details")
            for idx, doc in enumerate(data["documents"], 1):
                lines.append(f"### Document {idx}")
                lines.append(f"- **Source**: {doc.get('source', 'Unknown')}")
                lines.append(f"- **Metadata**: {doc.get('metadata', {})}")
                lines.append(f"- **Content Preview**: {doc.get('content', 'No content')}")
                lines.append("")
        
        # Tool Calls Section
        if data.get("tool_calls"):
            lines.append("## Tool Calls")
            for idx, tool_call in enumerate(data["tool_calls"], 1):
                lines.append(f"### Tool Call {idx}")
                if isinstance(tool_call, dict):
                    for key, value in tool_call.items():
                        lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
                else:
                    lines.append(f"- **Tool**: {tool_call}")
                lines.append("")
        
        # Document Sources Summary
        if data.get("document_sources"):
            lines.append("## Document Sources Summary")
            for source_type, count in data["document_sources"].items():
                lines.append(f"- **{source_type.replace('_', ' ').title()}**: {count} documents")
            lines.append("")
        
        # Cross-Reference Analysis
        if data.get("cross_reference_analysis"):
            lines.append("## Cross-Reference Analysis")
            analysis = data["cross_reference_analysis"]
            for key, value in analysis.items():
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            lines.append("")
        
        # Performance Metrics
        if data.get("performance_metrics"):
            lines.append("## Performance Metrics")
            metrics = data["performance_metrics"]
            for key, value in metrics.items():
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            lines.append("")
        
        # Session Information
        if data.get("session_info"):
            lines.append("## Session Information")
            session_info = data["session_info"]
            for key, value in session_info.items():
                lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            lines.append("")
        
        # Processing Summary
        lines.append("## Processing Information")
        lines.append(f"- **Documents Used**: {data.get('documents_used', 0)}")
        lines.append(f"- **Routing Decision**: {data.get('routing_decision', 'Unknown')}")
        lines.append("")
        
        return "\n".join(lines)
    
    # Handle old format or detailed graph output
    answer_data = data.get("answer", {})
    
    # Check if answer_data has the detailed structure (messages, documents, etc.)
    if isinstance(answer_data, dict):
        if "messages" in answer_data:
            lines.append("## Messages")
            for i, msg in enumerate(answer_data["messages"], 1):
                if isinstance(msg, dict):
                    content = msg.get("content", "")
                    msg_type = msg.get("type", "unknown")
                    lines.append(f"### Message {i} ({msg_type})")
                    lines.append(content)
                elif hasattr(msg, 'content'):
                    # Handle LangChain message objects
                    lines.append(f"### Message {i}")
                    lines.append(f"- **Content**: {msg.content}")
                    if hasattr(msg, 'additional_kwargs') and msg.additional_kwargs:
                        lines.append(f"- **Additional**: {msg.additional_kwargs}")
                    if hasattr(msg, 'id'):
                        lines.append(f"- **ID**: {msg.id}")
                else:
                    lines.append(f"- {msg}")
                lines.append("")

        if "Intermediate_message" in answer_data:
            lines.append("## Intermediate Message")
            lines.append(answer_data["Intermediate_message"])
            lines.append("")

        if "documents" in answer_data:
            lines.append("## Documents")
            documents = answer_data["documents"]
            # Handle case where documents might be a string or other type
            if isinstance(documents, list):
                for i, doc in enumerate(documents, 1):
                    lines.append(f"### Document {i}")
                    if isinstance(doc, dict):
                        if "metadata" in doc:
                            lines.append(f"**Metadata:** {json.dumps(doc['metadata'], indent=2)}")
                        if "page_content" in doc:
                            lines.append(f"**Content:** {doc['page_content']}")
                        if "type" in doc:
                            lines.append(f"**Type:** {doc['type']}")
                    elif hasattr(doc, 'page_content'):
                        # Handle LangChain document objects
                        lines.append(f"**Content:** {doc.page_content}")
                        if hasattr(doc, 'metadata'):
                            lines.append(f"**Metadata:** {json.dumps(doc.metadata, indent=2)}")
                    else:
                        lines.append(str(doc))
                    lines.append("")
            else:
                # Handle non-list documents (string, etc.)
                lines.append(f"**Documents:** {str(documents)}")
                lines.append("")

        if "retry_count" in answer_data:
            lines.append("## Retry Count")
            lines.append(str(answer_data["retry_count"]))
            lines.append("")

        if "tool_calls" in answer_data:
            lines.append("## Tool Calls")
            for i, call in enumerate(answer_data["tool_calls"], 1):
                if isinstance(call, dict):
                    lines.append(f"### Tool Call {i}")
                    lines.append(f"- **Tool**: {call.get('tool', 'Unknown')}")
                    if "input" in call:
                        lines.append(f"- **Input**: {json.dumps(call.get('input'), indent=2)}")
                    if "output" in call:
                        lines.append(f"- **Output**: {json.dumps(call.get('output'), indent=2)}")
                else:
                    lines.append(f"### Tool Call {i}")
                    lines.append(str(call))
                lines.append("")

        # Add additional fields that might be present in the detailed format
        if "vectorstore_searched" in answer_data:
            lines.append("## Search Information")
            lines.append(f"- **Vectorstore Searched**: {answer_data.get('vectorstore_searched', False)}")
            lines.append(f"- **Web Searched**: {answer_data.get('web_searched', False)}")
            lines.append(f"- **Vectorstore Quality**: {answer_data.get('vectorstore_quality', 'Unknown')}")
            lines.append("")
        
        if "cross_reference_analysis" in answer_data:
            lines.append("## Cross-Reference Analysis")
            cross_ref = answer_data["cross_reference_analysis"]
            if isinstance(cross_ref, dict):
                for key, value in cross_ref.items():
                    lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            else:
                lines.append(str(cross_ref))
            lines.append("")

        if "routing_memory" in answer_data:
            lines.append("## Routing Memory")
            routing = answer_data["routing_memory"]
            if isinstance(routing, dict):
                for key, value in routing.items():
                    if key == "scores" and isinstance(value, dict):
                        lines.append(f"- **Scores**:")
                        for score_type, score_val in value.items():
                            lines.append(f"  - {score_type}: {score_val:.3f}")
                    else:
                        lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            lines.append("")

        if "performance_metrics" in answer_data:
            lines.append("## Performance Metrics")
            perf = answer_data["performance_metrics"]
            if isinstance(perf, dict):
                for key, value in perf.items():
                    lines.append(f"- **{key.replace('_', ' ').title()}**: {value}")
            lines.append("")

    return "\n".join(lines)


def format_ingestion_output(data: dict) -> str:
    """Format ingestion response into Markdown with clear logs."""
    lines = []
    answer_data = data.get("answer", {})

    if "request" in answer_data:
        lines.append("## Request")
        lines.append(answer_data["request"])
        lines.append("")

    if "logs" in answer_data:
        lines.append("## Ingestion Logs")
        for i, log in enumerate(answer_data["logs"], 1):
            lines.append(f"{i}. {log}")
        lines.append("")

    lines.append("## File Information")
    lines.append(f"- Source: {answer_data.get('source')}")
    lines.append(f"- File Name: {answer_data.get('file_name')}")
    lines.append(f"- Space Key: {answer_data.get('space_key')}")
    lines.append(f"- Ticket ID: {answer_data.get('ticket_id')}")
    lines.append(f"- File URL: {answer_data.get('file_url')}")
    lines.append("")

    return "\n".join(lines)


def log_response(payload: dict, data: dict, folder: str = "responses") -> None:
    """Save formatted response to markdown log."""
    os.makedirs(folder, exist_ok=True)

    now = datetime.datetime.now()
    filename = now.strftime("%Y-%m-%d_%H-%M-%S") + ".md"
    filepath = os.path.join(folder, filename)

    # Decide which formatter to use
    if "logs" in data.get("answer", {}):
        content = format_ingestion_output(data)
    else:
        content = format_graph_output(data)

    # Create header with more details
    header_lines = [
        "# API Response Report",
        "="*50,
        f"Generated: {now.strftime('%Y-%m-%d %H:%M:%S')}",
        f"Query: {payload.get('query', 'Unknown')}",
        f"User ID: {payload.get('user_id', 'Unknown')}",
    ]
    
    # Add extra inputs if present
    if payload.get('extra_inputs'):
        header_lines.append(f"Extra Inputs: {json.dumps(payload['extra_inputs'], indent=2)}")
    
    # Add comprehensive response summary
    if isinstance(data.get('answer'), str):
        citations_count = len(data.get('citations', []))
        tool_calls_count = len(data.get('tool_calls', []))
        doc_sources = data.get('document_sources', {})
        
        header_lines.extend([
            "",
            "## Response Summary",
            f"- Documents Used: {data.get('documents_used', 0)}",
            f"- Citations Available: {citations_count}",
            f"- Tool Calls Made: {tool_calls_count}",
            f"- Routing Decision: {data.get('routing_decision', 'Unknown')}",
            f"- Session ID: {data.get('session_info', {}).get('session_id', 'Unknown')}",
            f"- Conversation Length: {data.get('session_info', {}).get('conversation_length', 0)}",
            f"- Cache Hit Rate: {data.get('session_info', {}).get('cache_hit_rate', 0):.2%}",
        ])
        
        # Add document source breakdown if available
        if doc_sources:
            header_lines.append("- Document Sources:")
            for source_type, count in doc_sources.items():
                if count > 0:
                    header_lines.append(f"  - {source_type.replace('_', ' ').title()}: {count}")
        
        # Add performance metrics summary
        perf_metrics = data.get('performance_metrics', {})
        if perf_metrics:
            header_lines.append("- Processing Flags:")
            for metric, value in perf_metrics.items():
                if isinstance(value, bool) and value:
                    header_lines.append(f"  - {metric.replace('_', ' ').title()}: ✓")
                elif not isinstance(value, bool) and value not in [0, 'none', None]:
                    header_lines.append(f"  - {metric.replace('_', ' ').title()}: {value}")
    
    header_lines.extend([
        "="*50,
        ""
    ])

    md_content = "\n".join(header_lines) + content

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(md_content)
    
    print(f"📝 Response logged to: {filepath}")
