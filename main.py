import os
import base64
import logging
from typing import Optional

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

from extractor import MarketingAssetExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def _require_env(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(f"Missing required env var: {name}")
    return val


SUPABASE_URL       = _require_env("SUPABASE_URL")
SUPABASE_KEY       = _require_env("SUPABASE_SERVICE_KEY")
ANTHROPIC_API_KEY  = _require_env("ANTHROPIC_API_KEY")
WEBHOOK_API_KEY    = _require_env("WEBHOOK_API_KEY")
AUTOCOPY_SCHEMA    = os.environ.get("AUTOCOPY_SCHEMA", "autocopy")

app = FastAPI(title="AutoCopy Asset Parser", version="3.0.0")
extractor = MarketingAssetExtractor(
    supabase_url=SUPABASE_URL,
    supabase_key=SUPABASE_KEY,
    anthropic_api_key=ANTHROPIC_API_KEY,
    schema=AUTOCOPY_SCHEMA,
)


class ParseRequest(BaseModel):
    filename: str
    file_b64: str
    brand: str
    state: str
    category: Optional[str] = None
    product: Optional[str] = None
    asset_type: Optional[str] = None
    source_path: Optional[str] = None
    state_list: Optional[list[str]] = None
    is_multistate: bool = False
    auto_provision: bool = True


class ParseResponse(BaseModel):
    status: str
    filename: str
    silo_table: Optional[str] = None
    brand: Optional[str] = None
    state: Optional[str] = None
    pages: int = 0
    blocks_saved: int = 0
    error: Optional[str] = None


@app.get("/")
def health():
    return {"status": "ok", "service": "autocopy-parser", "schema": AUTOCOPY_SCHEMA, "model": "claude-haiku-4-5"}


@app.get("/health")
def health2():
    return {"status": "ok"}


@app.post("/parse", response_model=ParseResponse)
def parse(payload: ParseRequest, x_api_key: str = Header(default="")):
    if x_api_key != WEBHOOK_API_KEY:
        raise HTTPException(status_code=401, detail="invalid api key")

    if not payload.file_b64:
        raise HTTPException(status_code=400, detail="file_b64 is empty")

    try:
        pdf_bytes = base64.b64decode(payload.file_b64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"invalid base64: {e}")

    if len(pdf_bytes) < 100 or not pdf_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="payload is not a PDF")

    logger.info(
        "parse start: filename=%r brand=%r state=%r category=%r product=%r size=%d",
        payload.filename, payload.brand, payload.state,
        payload.category, payload.product, len(pdf_bytes),
    )

    try:
        result = extractor.process_pdf_bytes(
            pdf_bytes=pdf_bytes,
            filename=payload.filename,
            brand=payload.brand,
            state=payload.state,
            category=payload.category,
            product=payload.product,
            asset_type=payload.asset_type,
            source_path=payload.source_path,
            state_list=payload.state_list,
            is_multistate=payload.is_multistate,
            auto_provision=payload.auto_provision,
        )
        return ParseResponse(
            status="ok",
            filename=payload.filename,
            silo_table=result.get("silo_table"),
            brand=result.get("brand"),
            state=result.get("state"),
            pages=result.get("pages", 0),
            blocks_saved=result.get("blocks_saved", 0),
        )
    except Exception as e:
        logger.exception("parse failed")
        return ParseResponse(
            status="error",
            filename=payload.filename,
            brand=payload.brand,
            state=payload.state,
            error=str(e),
        )
