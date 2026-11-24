"""Tests for the customer support API and knowledge retrieval."""

from types import SimpleNamespace

import pytest

import main
from knowledge_base import KnowledgeBaseError, load_knowledge_base, search_knowledge_base


@pytest.fixture
def client(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    main.app.config.update(TESTING=True)
    with main.app.test_client() as test_client:
        yield test_client


def test_root_and_health_endpoints(client):
    root_response = client.get("/")
    health_response = client.get("/health")

    assert root_response.status_code == 200
    assert "/ask" in root_response.json["endpoints"]
    assert health_response.status_code == 200
    assert health_response.json == {"status": "ok"}


def test_ask_returns_retrieved_answer_and_sources_without_api_key(client):
    response = client.post(
        "/ask", json={"message": "How can AI agents use business documents?"}
    )

    assert response.status_code == 200
    assert response.json["success"] is True
    assert response.json["question"] == "How can AI agents use business documents?"
    assert "knowledge base" in response.json["answer"].lower()
    assert response.json["sources"]
    assert response.json["ai_enabled"] is False


def test_ask_returns_fallback_when_no_records_match(client):
    response = client.post("/ask", json={"message": "What is the moon made of?"})

    assert response.status_code == 200
    assert response.json["success"] is True
    assert response.json["sources"] == []
    assert response.json["ai_enabled"] is False
    assert "couldn't find" in response.json["answer"]


def test_webhook_uses_same_answer_contract(client):
    response = client.post(
        "/webhook", json={"message": "What is retrieval-augmented generation?"}
    )

    assert response.status_code == 200
    assert response.json["success"] is True
    assert response.json["sources"]
    assert response.json["ai_enabled"] is False


@pytest.mark.parametrize(
    ("body", "expected_error"),
    [
        ({}, "non-empty string"),
        ({"message": "   "}, "non-empty string"),
        ({"message": 123}, "non-empty string"),
        ({"message": "x" * (main.MAX_MESSAGE_LENGTH + 1)}, "at most"),
        ([], "JSON object"),
    ],
)
def test_ask_rejects_invalid_payloads(client, body, expected_error):
    response = client.post("/ask", json=body)

    assert response.status_code == 400
    assert response.json["success"] is False
    assert expected_error in response.json["error"]


def test_ask_rejects_invalid_json(client):
    response = client.post(
        "/ask", data="{not json", content_type="application/json"
    )

    assert response.status_code == 400
    assert response.json["error"] == "Request body must contain valid JSON."


def test_ask_rejects_non_json_content_type(client):
    response = client.post("/ask", data="message=hello")

    assert response.status_code == 400
    assert response.json["error"] == "Content-Type must be application/json."


def test_ai_failure_falls_back_without_exposing_provider_error(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    def fail_generation(question, documents):
        raise RuntimeError("sensitive provider details")

    monkeypatch.setattr(main, "_generate_ai_answer", fail_generation)
    response = client.post("/ask", json={"message": "What is RAG?"})

    assert response.status_code == 200
    assert response.json["ai_enabled"] is False
    assert "retrieves relevant information" in response.json["answer"]
    assert "sensitive provider details" not in response.get_data(as_text=True)
    assert "test-key" not in response.get_data(as_text=True)


def test_ai_answer_is_used_when_generation_succeeds(client, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    class FakeCompletions:
        def create(self, **kwargs):
            assert kwargs["model"] == main.OPENAI_MODEL
            assert "Business context:" in kwargs["messages"][1]["content"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        message=SimpleNamespace(content="Generated grounded answer.")
                    )
                ]
            )

    class FakeOpenAI:
        def __init__(self, timeout):
            assert timeout == 20.0
            self.chat = SimpleNamespace(completions=FakeCompletions())

    monkeypatch.setattr(main, "OpenAI", FakeOpenAI)
    response = client.post("/ask", json={"message": "What is RAG?"})

    assert response.status_code == 200
    assert response.json["answer"] == "Generated grounded answer."
    assert response.json["ai_enabled"] is True


def test_missing_knowledge_base_returns_service_error(client, monkeypatch, tmp_path):
    monkeypatch.setattr(main, "KNOWLEDGE_BASE_PATH", tmp_path / "missing.json")

    response = client.post("/ask", json={"message": "Any question"})

    assert response.status_code == 500
    assert response.json == {
        "success": False,
        "error": "The knowledge base is unavailable.",
    }


def test_invalid_knowledge_base_json_raises_clear_error(tmp_path):
    invalid_file = tmp_path / "invalid.json"
    invalid_file.write_text("{", encoding="utf-8")

    with pytest.raises(KnowledgeBaseError, match="invalid JSON"):
        load_knowledge_base(invalid_file)


def test_keyword_search_prioritizes_title_and_tags():
    records = [
        {
            "id": "content",
            "title": "General support",
            "content": "Use RAG to retrieve information.",
            "tags": [],
        },
        {
            "id": "tag",
            "title": "Knowledge source",
            "content": "A collection of approved records.",
            "tags": ["RAG"],
        },
        {
            "id": "title",
            "title": "RAG retrieval",
            "content": "Find approved records.",
            "tags": [],
        },
    ]

    results = search_knowledge_base("RAG", records)

    assert [record["id"] for record in results] == ["title", "tag", "content"]
