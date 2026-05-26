"""
Marketing asset extractor v2 — schema-isolated, silo-routed.

Differences from v1:
  - Routes inserts to autocopy.<table_name> based on (brand, state) registry lookup
  - Auto-provisions the silo via autocopy.create_silo() if it doesn't exist
  - Sends Content-Profile: autocopy on every Supabase REST call
  - Stamps brand, state, category, product, asset_type, source_path on each row
"""

import json
import time
import warnings
import logging
from io import BytesIO
from typing import Dict, List, Any, Optional

import pdfplumber
import openai
import requests

warnings.filterwarnings("ignore", category=UserWarning, module="pdfminer")
logger = logging.getLogger(__name__)


class MarketingAssetExtractor:
    def __init__(self, supabase_url: str, supabase_key: str, openai_api_key: str, schema: str = "autocopy"):
        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_key = supabase_key
        self.schema = schema
        self.base_headers = {
            "apikey": supabase_key,
            "Authorization": f"Bearer {supabase_key}",
            "Content-Type": "application/json",
            "Content-Profile": schema,
            "Accept-Profile": schema,
            "Prefer": "return=representation",
        }
        self.openai_client = openai.OpenAI(api_key=openai_api_key)
        # In-process cache: (brand, state) -> table_name
        self._silo_cache: Dict[str, str] = {}

    # ─────────── Silo routing ───────────

    def _silo_key(self, brand: str, state: str) -> str:
        return f"{brand}::{state}"

    def lookup_silo(self, brand: str, state: str) -> Optional[str]:
        key = self._silo_key(brand, state)
        if key in self._silo_cache:
            return self._silo_cache[key]
        url = f"{self.supabase_url}/rest/v1/silo_registry"
        params = {
            "select": "table_name",
            "brand": f"eq.{brand}",
            "state": f"eq.{state}",
            "limit": 1,
        }
        r = requests.get(url, headers=self.base_headers, params=params, timeout=15)
        if r.status_code != 200:
            logger.error(f"silo lookup failed: {r.status_code} {r.text[:300]}")
            return None
        rows = r.json() or []
        if not rows:
            return None
        table = rows[0].get("table_name")
        if table:
            self._silo_cache[key] = table
        return table

    def provision_silo(
        self,
        brand: str,
        state: str,
        is_multistate: bool = False,
        state_list: Optional[List[str]] = None,
    ) -> Optional[str]:
        url = f"{self.supabase_url}/rest/v1/rpc/create_silo"
        body = {
            "p_brand": brand,
            "p_state": state,
            "p_is_multistate": is_multistate,
            "p_state_list": state_list,
        }
        r = requests.post(url, headers=self.base_headers, json=body, timeout=30)
        if r.status_code not in (200, 201):
            logger.error(f"create_silo failed: {r.status_code} {r.text[:400]}")
            return None
        table = r.json()
        if isinstance(table, str) and table:
            self._silo_cache[self._silo_key(brand, state)] = table
            logger.info(f"provisioned silo: {table}")
            return table
        return None

    def get_or_create_silo(
        self,
        brand: str,
        state: str,
        is_multistate: bool = False,
        state_list: Optional[List[str]] = None,
        auto_provision: bool = True,
    ) -> Optional[str]:
        existing = self.lookup_silo(brand, state)
        if existing:
            return existing
        if not auto_provision:
            return None
        return self.provision_silo(brand, state, is_multistate, state_list)

    # ─────────── Embeddings ───────────

    def generate_embedding(self, text: str) -> Optional[List[float]]:
        try:
            clean = text.replace("\n", " ").replace("\r", " ").strip()
            if not clean:
                return None
            if len(clean) > 8000:
                clean = clean[:8000]
            r = self.openai_client.embeddings.create(
                model="text-embedding-3-large",
                input=clean,
            )
            return r.data[0].embedding
        except Exception as e:
            logger.error(f"embedding failed: {e}")
            return None

    # ─────────── PDF parsing ───────────

    def extract_text_and_tables(self, pdf_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        pages_data: List[Dict[str, Any]] = []
        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    raw_tables = page.extract_tables() or []
                    table_rows: List[Dict[str, str]] = []
                    for table in raw_tables:
                        if not table or len(table) < 2:
                            continue
                        headers = [str(h).strip() if h else "" for h in table[0]]
                        for row in table[1:]:
                            if len(row) != len(headers):
                                continue
                            row_dict = {}
                            for i, h in enumerate(headers):
                                if h:
                                    row_dict[h] = str(row[i]).strip() if row[i] else ""
                            if row_dict:
                                table_rows.append(row_dict)
                    pages_data.append({
                        "page_number": page_num,
                        "text": text,
                        "tables": table_rows,
                        "source_document": filename,
                    })
        except Exception as e:
            logger.error(f"pdf read failed for {filename}: {e}")
        return pages_data

    # ─────────── OpenAI block extractors ───────────

    def _openai_json(self, system: str, user: str) -> Optional[Dict[str, Any]]:
        try:
            r = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.1,
                max_tokens=2000,
                response_format={"type": "json_object"},
            )
            raw = r.choices[0].message.content or "{}"
            cleaned = raw.replace("```json", "").replace("```", "").strip()
            return json.loads(cleaned)
        except Exception as e:
            logger.error(f"openai json call failed: {e}")
            return None

    def extract_pricing_data(self, tables: List[Dict], text: str, brand: str) -> List[Dict]:
        if not tables:
            return []
        prompt = (
            f"Extract ALL pricing data from these tables. Return ONLY valid JSON.\n\n"
            f"Brand: {brand}\n\nTables: {json.dumps(tables, indent=2)}\n\n"
            f"Context: {text[:500]}\n\n"
            'Return: {"products":[{"product_name":"...","sku":"...","thc_percentage":"...","price":"...",'
            '"case_pack":"...","unit_size":"...","product_format":"...","strain_type":"...","key_features":[]}]}'
        )
        data = self._openai_json("Extract pricing data. Return only valid JSON.", prompt)
        out = []
        for product in (data or {}).get("products", []) or []:
            out.append({
                "content_type": "pricing_data",
                "brand": brand,
                "product_name": product.get("product_name"),
                "sku": product.get("sku"),
                "thc_percentage": product.get("thc_percentage"),
                "price": product.get("price"),
                "case_pack": product.get("case_pack"),
                "unit_size": product.get("unit_size"),
                "product_format": product.get("product_format"),
                "strain_type": product.get("strain_type"),
                "key_features": product.get("key_features", []),
                "extraction_keywords": [brand, product.get("product_name", ""), "pricing", "wholesale"],
            })
        return out

    def extract_technical_content(self, text: str, brand: str) -> List[Dict]:
        if len(text) < 200 or not any(
            k in text.lower()
            for k in ["extract", "process", "manufacturing", "cultivation", "thc", "cbd", "terpene", "strain"]
        ):
            return []
        prompt = (
            f"Extract technical specifications and process details. Return ONLY valid JSON.\n\n"
            f"Brand: {brand}\n\nText: {text[:3000]}\n\n"
            'Return: {"technical_blocks":[{"content_type":"technical_specifications","title":"...",'
            '"extraction_method":"...","thc_range":"...","cbd_range":"...","terpenes":[],'
            '"cultivation_method":"...","equipment":[],"process_details":"...","key_features":[],'
            '"extraction_keywords":[]}]}\n\nIf none: {"technical_blocks":[]}'
        )
        data = self._openai_json("Extract technical specifications. Return only valid JSON.", prompt)
        out = []
        for block in (data or {}).get("technical_blocks", []) or []:
            block["brand"] = brand
            out.append(block)
        return out

    def extract_product_details(self, text: str, brand: str) -> List[Dict]:
        if len(text) < 200 or not any(
            k in text.lower() for k in ["product", "strain", "flavor", "effect", "available", "line"]
        ):
            return []
        prompt = (
            f"Extract product details and descriptions. Return ONLY valid JSON.\n\n"
            f"Brand: {brand}\n\nText: {text[:3000]}\n\n"
            'Return: {"product_blocks":[{"content_type":"product_details","product_line":"...",'
            '"product_name":"...","description":"...","flavors":[],"effects":[],"strains":[],'
            '"sizes":[],"formats":[],"key_features":[],"target_use":"...","extraction_keywords":[]}]}\n\n'
            'If none: {"product_blocks":[]}'
        )
        data = self._openai_json("Extract product details. Return only valid JSON.", prompt)
        out = []
        for block in (data or {}).get("product_blocks", []) or []:
            block["brand"] = brand
            out.append(block)
        return out

    def extract_brand_content(self, text: str, brand: str) -> List[Dict]:
        if len(text) < 200 or not any(
            k in text.lower() for k in ["mission", "story", "quality", "commitment", "experience", "craft"]
        ):
            return []
        prompt = (
            f"Extract brand messaging and positioning. Return ONLY valid JSON.\n\n"
            f"Brand: {brand}\n\nText: {text[:3000]}\n\n"
            'Return: {"brand_blocks":[{"content_type":"brand_messaging","title":"...","brand_story":"...",'
            '"positioning":"...","values":[],"quality_claims":[],"target_audience":"...",'
            '"competitive_advantages":[],"extraction_keywords":[]}]}\n\nIf none: {"brand_blocks":[]}'
        )
        data = self._openai_json("Extract brand messaging. Return only valid JSON.", prompt)
        out = []
        for block in (data or {}).get("brand_blocks", []) or []:
            block["brand"] = brand
            out.append(block)
        return out

    def extract_all_content(self, page_data: Dict, brand: str) -> List[Dict]:
        blocks: List[Dict] = []
        if page_data.get("tables"):
            blocks.extend(self.extract_pricing_data(page_data["tables"], page_data["text"], brand))
        text = page_data.get("text") or ""
        if text and len(text) > 100:
            blocks.extend(self.extract_technical_content(text, brand))
            blocks.extend(self.extract_product_details(text, brand))
            blocks.extend(self.extract_brand_content(text, brand))
        return blocks

    # ─────────── Content text building ───────────

    def build_content_text(self, block: Dict) -> str:
        parts = []
        if block.get("brand"):
            parts.append(f"Brand: {block['brand']}")
        ct = block.get("content_type", "")
        if ct == "pricing_data":
            if block.get("product_name"):    parts.append(f"Product: {block['product_name']}")
            if block.get("unit_size"):       parts.append(f"Size: {block['unit_size']}")
            if block.get("product_format"):  parts.append(f"Format: {block['product_format']}")
            if block.get("thc_percentage"):  parts.append(f"THC: {block['thc_percentage']}")
            if block.get("price"):           parts.append(f"Wholesale Price: ${block['price']}")
            if block.get("case_pack"):       parts.append(f"Case Pack: {block['case_pack']}")
            if block.get("strain_type"):     parts.append(f"Type: {block['strain_type']}")
        elif ct == "technical_specifications":
            if block.get("title"):              parts.append(block["title"])
            if block.get("process_details"):    parts.append(block["process_details"])
            if block.get("extraction_method"):  parts.append(f"Extraction: {block['extraction_method']}")
            if block.get("thc_range"):          parts.append(f"THC: {block['thc_range']}")
            if block.get("terpenes"):           parts.append(f"Terpenes: {', '.join(block['terpenes'])}")
        elif ct == "product_details":
            if block.get("product_line"):  parts.append(f"Product Line: {block['product_line']}")
            if block.get("product_name"):  parts.append(f"Product: {block['product_name']}")
            if block.get("description"):   parts.append(block["description"])
            if block.get("flavors"):       parts.append(f"Flavors: {', '.join(block['flavors'])}")
            if block.get("effects"):       parts.append(f"Effects: {', '.join(block['effects'])}")
        elif ct == "brand_messaging":
            if block.get("title"):        parts.append(block["title"])
            if block.get("brand_story"):  parts.append(block["brand_story"])
            if block.get("positioning"):  parts.append(block["positioning"])
        if block.get("key_features"):
            parts.append(f"Features: {', '.join(block['key_features'])}")
        return ". ".join(parts)

    def build_technical_data(self, block: Dict) -> Dict:
        if block.get("content_type") != "technical_specifications":
            return {}
        return {
            "extraction_method": block.get("extraction_method"),
            "thc_range": block.get("thc_range"),
            "cbd_range": block.get("cbd_range"),
            "terpenes": block.get("terpenes", []),
            "cultivation_method": block.get("cultivation_method"),
            "equipment": block.get("equipment", []),
            "process_details": block.get("process_details"),
        }

    def build_product_data(self, block: Dict) -> Dict:
        ct = block.get("content_type")
        if ct == "pricing_data":
            return {
                "product_name": block.get("product_name"),
                "sku": block.get("sku"),
                "unit_size": block.get("unit_size"),
                "product_format": block.get("product_format"),
                "strain_type": block.get("strain_type"),
                "pricing": {"price": block.get("price"), "case_pack": block.get("case_pack")},
                "potency": {"thc_percentage": block.get("thc_percentage")},
            }
        if ct == "product_details":
            return {
                "product_line": block.get("product_line"),
                "product_name": block.get("product_name"),
                "description": block.get("description"),
                "flavors": block.get("flavors", []),
                "effects": block.get("effects", []),
                "strains": block.get("strains", []),
                "sizes": block.get("sizes", []),
                "formats": block.get("formats", []),
                "target_use": block.get("target_use"),
            }
        return {}

    def build_brand_data(self, block: Dict) -> Dict:
        if block.get("content_type") != "brand_messaging":
            return {}
        return {
            "brand_story": block.get("brand_story"),
            "positioning": block.get("positioning"),
            "values": block.get("values", []),
            "quality_claims": block.get("quality_claims", []),
            "target_audience": block.get("target_audience"),
            "competitive_advantages": block.get("competitive_advantages", []),
        }

    def extract_cannabinoids(self, block: Dict) -> List[str]:
        out = []
        if block.get("thc_percentage"): out.append(f"THC: {block['thc_percentage']}")
        if block.get("cbd_percentage"): out.append(f"CBD: {block['cbd_percentage']}")
        if block.get("thc_range"):      out.append(f"THC: {block['thc_range']}")
        if block.get("cbd_range"):      out.append(f"CBD: {block['cbd_range']}")
        return out

    # ─────────── Silo-aware DB write ───────────

    def save_to_silo(
        self,
        silo_table: str,
        blocks: List[Dict],
        page_data: Dict,
        *,
        brand: str,
        state: str,
        category: Optional[str],
        product: Optional[str],
        asset_type: Optional[str],
        source_path: Optional[str],
        state_list: Optional[List[str]],
    ) -> int:
        saved = 0
        url = f"{self.supabase_url}/rest/v1/{silo_table}"
        for block in blocks:
            content_text = self.build_content_text(block)
            if not content_text or len(content_text) < 20:
                continue
            content_embedding = self.generate_embedding(content_text)
            if not content_embedding:
                continue
            row = {
                "source_document":               page_data["source_document"],
                "source_path":                   source_path,
                "page_number":                   page_data["page_number"],
                "brand":                         brand,
                "state":                         state,
                "state_list":                    state_list,
                "category":                      category,
                "product_name":                  block.get("product_name") or block.get("product_line") or product,
                "asset_type":                    asset_type,
                "content_type":                  block.get("content_type"),
                "content_text":                  content_text,
                "extraction_keywords":           block.get("extraction_keywords", []),
                "content_embedding":             content_embedding,
                "raw_text":                      (page_data.get("text") or "")[:5000],
                "comprehensive_technical_data":  self.build_technical_data(block),
                "comprehensive_product_data":    self.build_product_data(block),
                "comprehensive_brand_data":      self.build_brand_data(block),
                "all_cannabinoids":              self.extract_cannabinoids(block),
                "all_terpenes":                  block.get("terpenes", []),
                "all_flavors":                   block.get("flavors", []),
                "all_effects":                   block.get("effects", []),
                "all_product_sizes":             [block.get("unit_size")] if block.get("unit_size") else [],
                "all_strain_names":              block.get("strains", []),
                "all_equipment":                 block.get("equipment", []),
            }
            r = requests.post(url, headers=self.base_headers, json=row, timeout=30)
            if r.status_code in (200, 201):
                saved += 1
            else:
                logger.error(f"insert into {silo_table} failed: {r.status_code} {r.text[:300]}")
        return saved

    # ─────────── Entry point ───────────

    def process_pdf_bytes(
        self,
        pdf_bytes: bytes,
        filename: str,
        brand: str,
        state: str,
        category: Optional[str] = None,
        product: Optional[str] = None,
        asset_type: Optional[str] = None,
        source_path: Optional[str] = None,
        state_list: Optional[List[str]] = None,
        is_multistate: bool = False,
        auto_provision: bool = True,
    ) -> Dict[str, Any]:
        silo_table = self.get_or_create_silo(
            brand=brand, state=state,
            is_multistate=is_multistate, state_list=state_list,
            auto_provision=auto_provision,
        )
        if not silo_table:
            raise RuntimeError(f"no silo registered for ({brand!r}, {state!r}) and auto_provision is off or failed")

        pages = self.extract_text_and_tables(pdf_bytes, filename)
        if not pages:
            return {"brand": brand, "state": state, "silo_table": silo_table, "pages": 0, "blocks_saved": 0}

        total_saved = 0
        for page_data in pages:
            blocks = self.extract_all_content(page_data, brand)
            if blocks:
                total_saved += self.save_to_silo(
                    silo_table=silo_table, blocks=blocks, page_data=page_data,
                    brand=brand, state=state, category=category, product=product,
                    asset_type=asset_type, source_path=source_path, state_list=state_list,
                )
            time.sleep(0.5)

        return {
            "brand": brand, "state": state, "silo_table": silo_table,
            "pages": len(pages), "blocks_saved": total_saved,
        }
