"""Loading and keyword-based retrieval for the JSON knowledge base."""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence


class KnowledgeBaseError(Exception):
    """Raised when the knowledge-base file is unavailable or malformed."""


_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "can",
    "do",
    "does",
    "for",
    "how",
    "i",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "use",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def load_knowledge_base(path: Path) -> List[Dict[str, Any]]:
    """Load and validate knowledge records from a JSON file."""
    try:
        with path.open(encoding="utf-8") as file:
            records = json.load(file)
    except FileNotFoundError as exc:
        raise KnowledgeBaseError(f"Knowledge-base file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise KnowledgeBaseError(f"Knowledge-base file is invalid JSON: {path}") from exc
    except OSError as exc:
        raise KnowledgeBaseError(f"Could not read knowledge-base file: {path}") from exc

    if not isinstance(records, list):
        raise KnowledgeBaseError("Knowledge-base JSON must contain a list of records.")

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise KnowledgeBaseError(f"Knowledge-base record {index} must be an object.")
        if not all(
            isinstance(record.get(field), str) and record[field].strip()
            for field in ("id", "title", "content")
        ):
            raise KnowledgeBaseError(
                f"Knowledge-base record {index} needs non-empty id, title, and content."
            )
        tags = record.get("tags", [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise KnowledgeBaseError(
                f"Knowledge-base record {index} tags must be a list of strings."
            )

    return records


def _terms(text: str) -> set:
    return {
        token
        for token in _TOKEN_PATTERN.findall(text.lower())
        if token not in _STOP_WORDS
    }


def search_knowledge_base(
    question: str, records: Sequence[Dict[str, Any]], limit: int = 3
) -> List[Dict[str, Any]]:
    """Return records ranked by keyword overlap with title, tags, and content."""
    query_terms = _terms(question)
    if not query_terms:
        return []

    ranked = []
    for record in records:
        title_terms = _terms(record["title"])
        tag_terms = _terms(" ".join(record.get("tags", [])))
        content_terms = _terms(record["content"])
        score = (
            3 * len(query_terms & title_terms)
            + 2 * len(query_terms & tag_terms)
            + len(query_terms & content_terms)
        )
        if score:
            ranked.append((score, record))

    ranked.sort(key=lambda result: (-result[0], result[1]["id"]))
    return [record for _, record in ranked[:limit]]
