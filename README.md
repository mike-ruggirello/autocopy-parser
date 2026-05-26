# AutoCopy Asset Parser

FastAPI service that parses marketing PDFs and inserts extracted blocks into the Supabase `training_content` table. Designed to be called once per file by an n8n workflow.

## Endpoint

```
POST /parse
Headers:
  Content-Type: application/json
  x-api-key: <WEBHOOK_API_KEY>
Body:
  {
    "filename": "Almora_StrawberryLemonade_2026.pdf",
    "brand": "Almora",
    "file_b64": "<base64-encoded PDF bytes>"
  }
Response:
  { "status": "ok", "filename": "...", "brand": "Almora", "pages": 4, "blocks_saved": 12 }
```

## Environment variables

| Var | What |
|---|---|
| `SUPABASE_URL` | `https://uxzpfxlamulusbmsizuw.supabase.co` |
| `SUPABASE_SERVICE_KEY` | service-role key (bypasses RLS). Anon key works if RLS allows inserts. |
| `OPENAI_API_KEY` | OpenAI API key |
| `WEBHOOK_API_KEY` | any long random string. n8n sends this in `x-api-key` header. |

## Local run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then fill in real values
set -a; source .env; set +a
uvicorn main:app --reload
```

Test:
```bash
curl -s -X POST http://localhost:8000/parse \
  -H "Content-Type: application/json" \
  -H "x-api-key: $WEBHOOK_API_KEY" \
  -d "{\"filename\":\"test.pdf\",\"brand\":\"Almora\",\"file_b64\":\"$(base64 -i some.pdf)\"}"
```

## Render deploy

Render auto-detects `render.yaml`. Push this repo to GitHub, then in Render: New → Blueprint → connect repo. Fill in the four env vars in the Render dashboard.
