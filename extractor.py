"""
Marketing asset extractor v5 — Claude block extraction, campaign-level routing.

v5 changes:
  - Removed matcher webhook entirely (no SKU resolution)
  - Removed product_name column from silo rows
  - Bumped Claude max_tokens 2000 -> 8000 to avoid mid-sentence cutoffs
  - Campaign (folder name) is the primary grouping field
"""

import os
import re
import json
import time
import warnings
import logging
from io import BytesIO
from typing import Dict, List, Any, Optional

import pdfplumber
import anthropic
import requests

warnings.filterwarnings("ignore", category=UserWarning, module="pdfminer")
logger = logging.getLogger(__name__)

class MarketingAssetExtractor:
    def __init__(self, supabase_url: str, supabase_key: str, anthropic_api_key: str, schema: str = "autocopy"):
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
        self.anthropic_client = anthropic.Anthropic(api_key=anthropic_api_key)
        self.model = "claude-haiku-4-5-20251001"
        self._silo_cache: Dict[str, str] = {}

    # ─────────── Silo routing ───────────

    def _silo_key(self, brand: str, state: str) -> str:
        return f"{brand}::{state}"

    def lookup_silo(self, brand: str, state: str) -> Optional[str]:
        key = self._silo_key(brand, state)
        if key in self._silo_cache:
            return self._silo_cache[key]
        url = f"{self.supabase_url}/rest/v1/silo_registry"
        params = {"select": "table_name", "brand": f"eq.{brand}", "state": f"eq.{state}", "limit": 1}
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

    def provision_silo(self, brand, state, is_multistate=False, state_list=None):
        url = f"{self.supabase_url}/rest/v1/rpc/create_silo"
        body = {"p_brand": brand, "p_state": state,
                "p_is_multistate": is_multistate, "p_state_list": state_list}
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

    def get_or_create_silo(self, brand, state, is_multistate=False, state_list=None, auto_provision=True):
        existing = self.lookup_silo(brand, state)
        if existing:
            return existing
        if not auto_provision:
            return None
        return self.provision_silo(brand, state, is_multistate, state_list)

    # ─────────── PDF parsing ───────────

    def extract_text_and_tables(self, pdf_bytes: bytes, filename: str) -> List[Dict[str, Any]]:
        pages_data = []
        try:
            with pdfplumber.open(BytesIO(pdf_bytes)) as pdf:
                for page_num, page in enumerate(pdf.pages, 1):
                    text = page.extract_text() or ""
                    raw_tables = page.extract_tables() or []
                    table_rows = []
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

    # ─────────── Claude JSON extraction ───────────

    def _claude_json(self, system: str, user: str) -> Optional[Dict[str, Any]]:
        raw = ""
        try:
            msg = self.anthropic_client.messages.create(
                model=self.model,
                max_tokens=8000,
                system=system + " Output ONLY valid JSON. No prose before or after, no markdown code fences. Start with { and end with }.",
                messages=[{"role": "user", "content": user}],
            )
            for block in msg.content:
                if hasattr(block, "text"):
                    raw += block.text
            raw = (raw or "").strip().replace("```json", "").replace("```", "").strip()
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                raw = raw[start:end + 1]
            if not raw:
                return None
            return json.loads(raw)
        except json.JSONDecodeError as e:
            logger.error(f"claude returned non-JSON: {e}; raw={raw[:300]!r}")
            return None
        except Exception as e:
            logger.error(f"claude json call failed: {e}")
            return None

    def extract_pricing_data(self, tables, text, brand):
        if not tables:
            return []
        prompt = (f"Extract ALL pricing data. Return ONLY valid JSON.\n\nBrand: {brand}\n\n"
                  f"Tables: {json.dumps(tables, indent=2)}\n\nContext: {text[:500]}\n\n"
                  'Return: {"products":[{"product_name":"...","sku":"...","thc_percentage":"...","price":"...",'
                  '"case_pack":"...","unit_size":"...","product_format":"...","strain_type":"...","key_features":[]}]}')
        data = self._claude_json("Extract pricing data. Return only valid JSON.", prompt)
        return [{
            "content_type": "pricing_data", "brand": brand,
            "product_name": p.get("product_name"), "sku": p.get("sku"),
            "thc_percentage": p.get("thc_percentage"), "price": p.get("price"),
            "case_pack": p.get("case_pack"), "unit_size": p.get("unit_size"),
            "product_format": p.get("product_format"), "strain_type": p.get("strain_type"),
            "key_features": p.get("key_features", []),
            "extraction_keywords": [brand, p.get("product_name", ""), "pricing", "wholesale"],
        } for p in (data or {}).get("products", []) or []]

    def extract_technical_content(self, text, brand):
        if len(text) < 150 or not any(k in text.lower() for k in
            ["extract", "process", "manufactur", "cultivat", "thc", "cbd", "terpene", "strain", "potency", "mg"]):
            return []
        prompt = (f"Extract technical specs and process details. Return ONLY valid JSON.\n\nBrand: {brand}\n\n"
                  f"Text: {text[:3000]}\n\n"
                  'Return: {"technical_blocks":[{"content_type":"technical_specifications","title":"...",'
                  '"extraction_method":"...","thc_range":"...","cbd_range":"...","terpenes":[],"product_name":"...",'
                  '"cultivation_method":"...","equipment":[],"process_details":"...","key_features":[],"extraction_keywords":[]}]}\n\n'
                  'If none: {"technical_blocks":[]}')
        data = self._claude_json("Extract technical specifications. Return only valid JSON.", prompt)
        out = []
        for block in (data or {}).get("technical_blocks", []) or []:
            block["brand"] = brand
            out.append(block)
        return out

    def extract_product_details(self, text, brand):
        if len(text) < 100 or not any(k in text.lower() for k in
            ["product", "strain", "flavor", "effect", "available", "line", "tincture", "cart", "edible", "pre-roll", "flower"]):
            return []
        prompt = (f"Extract product details. Return ONLY valid JSON.\n\nBrand: {brand}\n\nText: {text[:3000]}\n\n"
                  'Return: {"product_blocks":[{"content_type":"product_details","product_line":"...","product_name":"...",'
                  '"description":"...","flavors":[],"effects":[],"strains":[],"sizes":[],"formats":[],"key_features":[],'
                  '"target_use":"...","extraction_keywords":[]}]}\n\nIf none: {"product_blocks":[]}')
        data = self._claude_json("Extract product details. Return only valid JSON.", prompt)
        out = []
        for block in (data or {}).get("product_blocks", []) or []:
            block["brand"] = brand
            out.append(block)
        return out

    def extract_brand_content(self, text, brand):
        if len(text) < 100 or not any(k in text.lower() for k in
            ["mission", "story", "quality", "commitment", "experience", "craft", "introducing", "welcome", "premium", "luxury"]):
            return []
        prompt = (f"Extract brand messaging. Return ONLY valid JSON.\n\nBrand: {brand}\n\nText: {text[:3000]}\n\n"
                  'Return: {"brand_blocks":[{"content_type":"brand_messaging","title":"...","brand_story":"...","product_name":null,'
                  '"positioning":"...","values":[],"quality_claims":[],"target_audience":"...","competitive_advantages":[],'
                  '"extraction_keywords":[]}]}\n\nIf none: {"brand_blocks":[]}')
        data = self._claude_json("Extract brand messaging. Return only valid JSON.", prompt)
        out = []
        for block in (data or {}).get("brand_blocks", []) or []:
            block["brand"] = brand
            out.append(block)
        return out

    def extract_all_content(self, page_data, brand):
        blocks = []
        if page_data.get("tables"):
            blocks.extend(self.extract_pricing_data(page_data["tables"], page_data["text"], brand))
        text = page_data.get("text") or ""
        if text and len(text) > 50:
            blocks.extend(self.extract_technical_content(text, brand))
            blocks.extend(self.extract_product_details(text, brand))
            blocks.extend(self.extract_brand_content(text, brand))
        return blocks

    # ─────────── Matcher integration ───────────

    def build_content_text(self, block):
        """Return only natural-language marketing prose from the page.
        No metadata labels (Brand:, Product:, Flavors: etc) - those live in JSON cols.
        No matched product_name (it gets overwritten post-extraction, contaminating training data).
        Pricing data has no creative copy - return empty so the min-length filter drops it.
        """
        parts = []
        ct = block.get("content_type", "")
        if ct == "brand_messaging":
            if block.get("title"):       parts.append(block["title"])
            if block.get("brand_story"): parts.append(block["brand_story"])
            if block.get("positioning"): parts.append(block["positioning"])
        elif ct == "product_details":
            if block.get("description"): parts.append(block["description"])
            if block.get("target_use"):  parts.append(block["target_use"])
        elif ct == "technical_specifications":
            if block.get("title"):           parts.append(block["title"])
            if block.get("process_details"): parts.append(block["process_details"])
        # pricing_data: structured numbers, no marketing prose -> empty -> dropped by min-length filter
        return " ".join(p.strip() for p in parts if p and str(p).strip())

    def build_technical_data(self, block):
        if block.get("content_type") != "technical_specifications": return {}
        return {"extraction_method": block.get("extraction_method"), "thc_range": block.get("thc_range"),
                "cbd_range": block.get("cbd_range"), "terpenes": block.get("terpenes", []),
                "cultivation_method": block.get("cultivation_method"), "equipment": block.get("equipment", []),
                "process_details": block.get("process_details")}

    def build_product_data(self, block):
        ct = block.get("content_type")
        if ct == "pricing_data":
            return {"product_name": block.get("product_name"), "sku": block.get("sku"),
                    "unit_size": block.get("unit_size"), "product_format": block.get("product_format"),
                    "strain_type": block.get("strain_type"),
                    "pricing": {"price": block.get("price"), "case_pack": block.get("case_pack")},
                    "potency": {"thc_percentage": block.get("thc_percentage")}}
        if ct == "product_details":
            return {"product_line": block.get("product_line"), "product_name": block.get("product_name"),
                    "description": block.get("description"), "flavors": block.get("flavors", []),
                    "effects": block.get("effects", []), "strains": block.get("strains", []),
                    "sizes": block.get("sizes", []), "formats": block.get("formats", []),
                    "target_use": block.get("target_use")}
        return {}

    def build_brand_data(self, block):
        if block.get("content_type") != "brand_messaging": return {}
        return {"brand_story": block.get("brand_story"), "positioning": block.get("positioning"),
                "values": block.get("values", []), "quality_claims": block.get("quality_claims", []),
                "target_audience": block.get("target_audience"),
                "competitive_advantages": block.get("competitive_advantages", [])}

    def extract_cannabinoids(self, block):
        out = []
        if block.get("thc_percentage"): out.append(f"THC: {block['thc_percentage']}")
        if block.get("cbd_percentage"): out.append(f"CBD: {block['cbd_percentage']}")
        if block.get("thc_range"):      out.append(f"THC: {block['thc_range']}")
        if block.get("cbd_range"):      out.append(f"CBD: {block['cbd_range']}")
        return out

    # ─────────── DB write ───────────

    def save_to_silo(self, silo_table, blocks, page_data, *, brand, state, category, product,
                     asset_type, source_path, state_list):
        saved = 0
        url = f"{self.supabase_url}/rest/v1/{silo_table}"
        for block in blocks:
            content_text = self.build_content_text(block)
            if not content_text or len(content_text) < 20:
                continue
            row = {
                "source_document":              page_data["source_document"],
                "source_path":                  source_path,
                "page_number":                  page_data["page_number"],
                "brand":                        brand,
                "state":                        state,
                "state_list":                   state_list,
                "category":                     category,
                "campaign":                     product,
                "asset_type":                   asset_type,
                "content_type":                 block.get("content_type"),
                "content_text":                 content_text,
                "extraction_keywords":          block.get("extraction_keywords", []),
                "raw_text":                     (page_data.get("text") or "")[:5000],
                "comprehensive_technical_data": self.build_technical_data(block),
                "comprehensive_product_data":   self.build_product_data(block),
                "comprehensive_brand_data":     self.build_brand_data(block),
                "all_cannabinoids":             self.extract_cannabinoids(block),
                "all_terpenes":                 block.get("terpenes", []),
                "all_flavors":                  block.get("flavors", []),
                "all_effects":                  block.get("effects", []),
                "all_product_sizes":            [block.get("unit_size")] if block.get("unit_size") else [],
                "all_strain_names":             block.get("strains", []),
                "all_equipment":                block.get("equipment", []),
            }
            r = requests.post(url, headers=self.base_headers, json=row, timeout=30)
            if r.status_code in (200, 201):
                saved += 1
            else:
                logger.error(f"insert into {silo_table} failed: {r.status_code} {r.text[:300]}")
        return saved

    # ─────────── Entry point ───────────

    def process_pdf_bytes(self, pdf_bytes, filename, brand, state, category=None, product=None,
                         asset_type=None, source_path=None, state_list=None, is_multistate=False,
                         auto_provision=True):
        silo_table = self.get_or_create_silo(brand=brand, state=state, is_multistate=is_multistate,
                                              state_list=state_list, auto_provision=auto_provision)
        if not silo_table:
            raise RuntimeError(f"no silo registered for ({brand!r}, {state!r})")

        pages = self.extract_text_and_tables(pdf_bytes, filename)
        if not pages:
            return {"brand": brand, "state": state, "silo_table": silo_table, "pages": 0, "blocks_saved": 0}

        total_saved = 0
        for page_data in pages:
            blocks = self.extract_all_content(page_data, brand)
            if not blocks:
                continue
            total_saved += self.save_to_silo(
                silo_table=silo_table, blocks=blocks, page_data=page_data,
                brand=brand, state=state, category=category, product=product,
                asset_type=asset_type, source_path=source_path, state_list=state_list,
            )
            time.sleep(0.3)

        return {"brand": brand, "state": state, "silo_table": silo_table,
                "pages": len(pages), "blocks_saved": total_saved}
