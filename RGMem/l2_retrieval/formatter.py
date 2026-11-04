# l2_retrieval/formatter.py

import html
from typing import Optional, List, Dict


def _escape(text: str) -> str:
    """Escapes Markdown content, specifically handling special characters."""
    # Using html.escape for general safety, though typical Markdown escapes (*, _, [, ]) are context-dependent.
    return html.escape(str(text))


def _format_timestamp(ts: str) -> str:
    """Truncates a full timestamp string to YYYY-MM-DD format."""
    if not ts or len(ts) < 10:
        return ""
    return ts[:10]


def _format_l0_recency_to_markdown(data: List[Dict]) -> str:
    """Formats L0 recency memories into a Markdown list."""
    if not data:
        return ""

    md_parts = ["## Recent Facts (L0)"]
    for fact in data:
        ts = _format_timestamp(fact.get('parent_created_at', ''))
        content = _escape(fact.get('content', ''))
        md_parts.append(f"- ({ts}) {content}")

    return "\n".join(md_parts)


def _format_l0_relevance_to_markdown(data: Dict[str, List[Dict]]) -> str:
    """Formats L0 relevance memories into a Markdown list."""
    if not data:
        return ""

    core_facts = data.get("core_facts", [])
    conclusions = data.get("conclusions", [])

    if not core_facts and not conclusions:
        return ""

    md_parts = ["## Relevant Memories (L0)"]
    if core_facts:
        md_parts.append("### Core Facts:")
        for fact in core_facts:
            ts = _format_timestamp(fact.get('parent_created_at', ''))
            content = _escape(fact.get('content', ''))
            md_parts.append(f"- ({ts}) {content}")

    if conclusions:
        md_parts.append("### Conclusions:")
        for conc in conclusions:
            ts = _format_timestamp(conc.get('parent_created_at', ''))
            content = _escape(conc.get('content', ''))
            md_parts.append(f"- ({ts}) {content}")

    return "\n".join(md_parts)


def _format_l1_kg_to_markdown(data: Dict, query: str) -> str:
    """Formats the L1 knowledge graph data into a hierarchical Markdown structure."""
    if not data or data.get("error"):
        return ""

    md_parts = [f"## Knowledge Graph Insights (L1 for query: '{_escape(query)}')"]
    seen_summaries = set()

    for entity_name, node_data in data.items():
        md_parts.append(f"\n### Entity: {_escape(entity_name)}")

        if node_data.get("summaries"):
            has_printed_summary_header = False
            for summary in node_data["summaries"]:
                summary_content = summary.get("summary", "")
                if summary_content and summary_content not in seen_summaries:
                    if not has_printed_summary_header:
                        md_parts.append("- **Summaries**:")
                        has_printed_summary_header = True
                    ts = _format_timestamp(summary.get("last_update", ""))
                    topic = _escape(summary.get("name", ""))
                    md_parts.append(f"  - (Topic: {topic}, Updated: {ts}) {_escape(summary_content)}")
                    seen_summaries.add(summary_content)

        if node_data.get("inferences"):
            md_parts.append("- **Inferences**:")
            for inference in node_data["inferences"]:
                ts = _format_timestamp(inference.get("last_update", ""))
                source = _escape(inference.get("source", ""))
                target = _escape(inference.get("target", ""))
                md_parts.append(
                    f"  - (On: '{source}'->'{target}', Updated: {ts}) {_escape(inference.get('inference', ''))}")

        if node_data.get("one_hop_details"):
            md_parts.append("- **Details**:")
            for rel in node_data["one_hop_details"]:
                rel_type = rel.get("relationship_type", "")
                source = _escape(rel.get("source", ""))
                target = _escape(rel.get("target", ""))
                md_parts.append(f"  - Relationship ({rel_type}): ({source}) -> ({target})")

                if rel.get("history"):
                    history_list = rel.get("history", [])
                    history_to_print = history_list
                    if rel_type == 'classification':
                        history_to_print = sorted(history_list, key=lambda x: x.get('timestamp', ''), reverse=True)[:1]

                    for hist in history_to_print:
                        ts = _format_timestamp(hist.get("timestamp", ""))
                        content = _escape(hist.get("relationship_content", ""))
                        md_parts.append(f"    - History ({ts}): {content}")

                if rel.get("conclusions"):
                    for conc in rel.get("conclusions"):
                        ts = _format_timestamp(conc.get("timestamp", ""))
                        content = _escape(conc.get("content", ""))
                        md_parts.append(f"    - Conclusion ({ts}): {content}")

    return "\n".join(md_parts)


def format_context_as_markdown(
        l0_recency_data: Optional[List[Dict]],
        l0_relevance_data: Optional[Dict[str, List[Dict]]],
        l1_kg_data: Optional[Dict],
        query: str
) -> str:
    """
    Dynamically and concisely assembles all retrieved raw data into a Markdown string.
    Markdown blocks for missing data types will not be generated.
    """
    md_parts = []

    recency_md = _format_l0_recency_to_markdown(l0_recency_data)
    if recency_md:
        md_parts.append(recency_md)

    relevance_l0_md = _format_l0_relevance_to_markdown(l0_relevance_data)
    if relevance_l0_md:
        md_parts.append(relevance_l0_md)

    relevance_l1_md = _format_l1_kg_to_markdown(l1_kg_data, query)
    if relevance_l1_md:
        md_parts.append(relevance_l1_md)

    if not md_parts:
        return "No relevant information found in memory."

    return "\n\n".join(md_parts)