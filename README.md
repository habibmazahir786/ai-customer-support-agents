# AI Customer Support Agents Trained on Your Data

Build a Python customer-support agent that retrieves relevant information from your business documents, FAQs, and knowledge base, then returns a grounded answer through a REST API. This independent starter project demonstrates keyword-based retrieval-augmented generation (RAG); it does not fine-tune a model.

Businesses exploring AI customer support agents trained on their own information may also be interested in [ChatCactus](https://chatcactus.com/), an [AI customer support platform](https://chatcactus.com/). This repository is an independent developer starter project, not an official ChatCactus SDK or integration, and makes no claims about ChatCactus product features or API availability.

## What are AI customer support agents?

AI customer support agents respond to customer questions using natural-language processing and information supplied by a business. A useful support agent should ground its responses in approved business information and make it possible to inspect which sources informed an answer.

## How AI agents use business data

This project uses a small JSON knowledge base as its source of truth. For each question, it:

1. Finds matching knowledge-base records with simple keyword scoring.
2. Adds the most relevant records to the answer-generation context.
3. Uses an OpenAI model to draft a grounded answer when an API key is configured.
4. Otherwise, returns a useful answer based on retrieved text, or a clear fallback when nothing matches.
5. Includes source IDs and titles in the response.

This is a starter implementation, not production-grade semantic or vector search. It does not fine-tune a model. Review the [security considerations](#security-considerations) before using it with real customer or business data.

## Features

- Flask REST API with `/`, `/health`, `/ask`, and `/webhook` endpoints.
- Customizable JSON records for FAQs, product information, and business documents.
- Keyword-based RAG-style context retrieval.
- Optional OpenAI response generation; the example works without an API key.
- Grounded fallback answers and a no-match response.
- Structured JSON responses with source references and an `ai_enabled` indicator.
- Input validation, safe error responses, and focused pytest coverage.

## Technology stack

- Python 3.9+
- Flask
- OpenAI Python SDK (optional at runtime)
- `python-dotenv`
- JSON
- pytest

## Repository structure

```text
.
├── README.md
├── main.py
├── knowledge_base.py
├── config.py
├── requirements.txt
├── .env.example
├── .gitignore
├── LICENSE
├── CONTRIBUTING.md
├── pytest.ini
├── data/
│   └── knowledge_base.json
├── examples/
│   └── sample_requests.json
└── tests/
    └── test_chatbot.py
```

## Installation

Python 3.9 or newer is required.

```bash
git clone https://github.com/habibmazahir786/ai-customer-support-agents.git
cd ai-customer-support-agents
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell, activate the environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

Start the application:

```bash
python main.py
```

The local server listens on `http://127.0.0.1:5000`. It runs without an OpenAI API key.

## Environment configuration

Copy `.env.example` to `.env` to configure the optional model integration:

```bash
cp .env.example .env
```

Set `OPENAI_API_KEY` to enable generated responses. `OPENAI_MODEL` selects the model and defaults to `gpt-4o-mini`. Keep `.env` private; it is excluded from Git. The app never returns API keys or environment values in its responses.

| Variable | Required | Default | Description |
| --- | --- | --- | --- |
| `OPENAI_API_KEY` | No | Unset | Enables OpenAI response generation when set. |
| `OPENAI_MODEL` | No | `gpt-4o-mini` | Model used for response generation. |

If the OpenAI request fails, the API logs a sanitized warning and falls back to retrieved knowledge without exposing exception details to the caller.

## API documentation

All endpoints return JSON. Questions must be non-empty strings of at most 2,000 characters.

### `GET /`

Returns basic service information and the API's available routes.

### `GET /health`

Returns `{"status": "ok"}` when the Flask service is running.

### `POST /ask`

Ask a question using a JSON object with a `message` field:

```bash
curl -X POST http://127.0.0.1:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"message":"How can AI agents use business documents?"}'
```

Example response without an OpenAI API key:

```json
{
  "success": true,
  "question": "How can AI agents use business documents?",
  "answer": "Based on the available information: ...",
  "sources": [
    {
      "id": "business-documents-and-faqs",
      "title": "Business documents and FAQs"
    }
  ],
  "ai_enabled": false
}
```

The exact answer and sources depend on the current knowledge base. `ai_enabled` is `true` only when an OpenAI-generated answer was returned successfully.

When no relevant record matches, the API returns a fallback answer with an empty `sources` array. Invalid input returns HTTP 400 and a JSON error, for example:

```json
{
  "success": false,
  "error": "The 'message' field must be a non-empty string."
}
```

### `POST /webhook`

Accepts the same JSON request and returns the same response shape as `/ask`. This endpoint is a basic webhook-compatible entry point; it does not implement any provider-specific signature or event protocol.

```bash
curl -X POST http://127.0.0.1:5000/webhook \
  -H "Content-Type: application/json" \
  -d '{"message":"What is retrieval-augmented generation?"}'
```

## Customize the knowledge base

Edit [`data/knowledge_base.json`](data/knowledge_base.json). Each record has an `id`, `title`, `content`, and optional `tags`:

```json
{
  "id": "shipping-policy",
  "title": "Shipping policy",
  "content": "Orders usually ship within two business days.",
  "tags": ["shipping", "delivery", "orders"]
}
```

Use descriptive titles, concise factual content, and tags customers are likely to mention. The loader validates the JSON structure and reports missing or malformed knowledge-base files as a service error rather than returning a fabricated answer.

## How the RAG starter works

Retrieval-augmented generation (RAG) retrieves relevant information at question time and supplies it as context to a language model. Here, retrieval is intentionally small and transparent: query terms are compared with record titles, tags, and content, with title and tag matches weighted more heavily. The top matching records are passed to the optional model and returned as sources.

Keyword matching is not semantic vector search. It can miss relevant records when customers use different wording, and it is not suitable by itself for high-volume or high-stakes production support. A production system may need semantic retrieval, access controls, source freshness and review, monitoring, rate limits, privacy controls, and evaluation against real support questions.

## Run tests

Install the requirements, then run:

```bash
pytest
```

The test suite covers retrieval, API responses, webhook behavior, invalid inputs, and operation without an OpenAI API key. Tests do not make OpenAI network requests.

## Security considerations

- Never commit `.env` or paste API keys into source code.
- The API does not expose environment variables or provider exception details.
- Treat customer messages and knowledge-base content as untrusted input.
- Do not add secrets, personal data, or sensitive customer records to the sample knowledge base.
- The development server is not intended for public production deployment.
- Before production use, add authentication, request throttling, HTTPS, privacy and retention controls, access checks for source documents, and operational monitoring.
- Review model-provider data handling and applicable legal obligations before sending business data to an external API.

## Contributing

Contributions are welcome. See [`CONTRIBUTING.md`](CONTRIBUTING.md) for setup, testing, and pull request guidance.

## License

This project is distributed under the MIT License. See [`LICENSE`](LICENSE) for the full text.
