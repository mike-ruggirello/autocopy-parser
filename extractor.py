"""
Marketing asset extractor v4 — Claude + matcher + no embeddings.

v4 changes:
  - Calls n8n matcher webhook per unique candidate per page
  - Drops cross-product mentions (matcher decision = 'cross_product')
  - Clears product_name on brand-level concepts
  - Stamps canonical subcategory from dash_products onto the category column
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
    def __init__(self, supabase_url: str, supabase_key: str, anthropic_api_key: str, schema: str = "autocopy",
                 matcher_url: Optional[str] = None):
        self.supabase_url = supabase_url.rstrip("/")
        self.supabase_key = supabase_key
        self.schema = schema
        self.matcher_url = matcher_url
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
                max_tokens=2000,
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

    def _call_matcher(self, brand: str, candidate: str, context: str, content_type: str,
                      primary_product: str = "") -> Dict[str, Any]:
        """POST to n8n matcher webhook. Returns the verdict dict or 'unknown' on error."""
        if not self.matcher_url:
            return {"decision": "unknown", "reason": "matcher_url not configured"}
        try:
            r = requests.post(
                self.matcher_url,
                json={
                    "brand": brand, "candidate": candidate, "context": context,
                    "content_type": content_type, "primary_product": primary_product,
                },
                timeout=30,
            )
            if r.status_code != 200:
                logger.error(f"matcher returned {r.status_code}: {r.text[:200]}")
                return {"decision": "unknown", "reason": f"http_{r.status_code}"}
            return r.json()
        except Exception as e:
            logger.error(f"matcher call failed: {e}")
            return {"decision": "unknown", "reason": str(e)}

    def match_blocks(self, blocks: List[Dict], brand: str, page_text: str,
                     filename: str = "", primary_product: str = "") -> List[Dict]:
        """
        For each block: call matcher, apply routing rule.
          matched      -> stamp canonical product/subcategory, keep
          brand_level  -> product_name=None, keep
          cross_product -> drop
          unknown      -> keep with original (fallback)
        Cache per (brand, candidate.lower()) to avoid duplicate matcher calls.
        Context sent to matcher: filename + full page text + primary_product, so matcher
        knows the document's primary subject and can detect cross-product mentions.
        """
        out = []
        cache: Dict[tuple, Dict] = {}
        page_ctx = f"DOCUMENT: {filename}\n\nPAGE CONTENT:\n{(page_text or '')[:3000]}"
        for block in blocks:
            candidate = (block.get("product_name") or block.get("product_line") or "").strip()
            content_type = block.get("content_type", "")
            if content_type == "brand_messaging" and not candidate:
                block["product_name"] = None
                block["_matcher_decision"] = "brand_messaging_no_candidate"
                out.append(block)
                continue
            if not candidate:
                block["_matcher_decision"] = "no_candidate"
                out.append(block)
                continue

            cache_key = (brand, candidate.lower())
            if cache_key not in cache:
                cache[cache_key] = self._call_matcher(
                    brand, candidate, page_ctx, content_type, primary_product=primary_product
                )
            verdict = cache[cache_key]
            decision = verdict.get("decision", "unknown")
            block["_matcher_decision"] = decision

            if decision == "matched":
                block["product_name"] = verdict.get("product_line") or verdict.get("product_name") or candidate
                block["_matched_inventory_id"] = verdict.get("inventory_id")
                block["_matched_subcategory"] = verdict.get("subcategory")
                block["_matched_category"] = verdict.get("category")
                out.append(block)
            elif decision == "brand_level" or decision == "no_match":
                # No real product matched. Don't fall back to the raw candidate text
                # (it's often a tagline fragment like "HYPED & HIGH" or "LOW"), set null.
                block["product_name"] = None
                out.append(block)
            elif decision == "cross_product":
                logger.info(f"dropped cross-product mention: brand={brand!r} candidate={candidate!r}")
                continue
            else:
                # unknown / matcher error: treat same as brand_level rather than leaking raw candidate
                block["product_name"] = None
                out.append(block)
        return out

    # ─────────── Content text building ───────────

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
            # Matched subcategory overrides path-derived category; fall back to path
            row_category = block.get("_matched_subcategory") or block.get("_matched_category") or category
            row = {
                "source_document":              page_data["source_document"],
                "source_path":                  source_path,
                "page_number":                  page_data["page_number"],
                "brand":                        brand,
                "state":                        state,
                "state_list":                   state_list,
                "category":                     row_category,
                "product_name":                 block.get("product_name"),
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

    def resolve_file_level_sku(self, filename: str, asset_type: str, brand: str,
                                 primary_product: str) -> Optional[Dict[str, Any]]:
        """For single-SKU files (sell sheets, postcards, table tents with specific variants),
        resolve the SKU once from the filename and reuse for every block. Returns None for
        multi-SKU files (training decks, partner guides, launch decks, 'combined' files) so
        the parser falls back to per-block matching.
        """
        if not filename:
            return None

        # Multi-SKU asset types - these intentionally cover multiple products per file
        multi_sku_asset_types = {"training deck", "partner guide", "launch deck", "education", "deck"}
        if asset_type and asset_type.lower().strip() in multi_sku_asset_types:
            return None

        # Strip extension, version suffixes, asset-type words from filename
        stem = filename.rsplit(".", 1)[0]
        # 'Combined' files cover multiple SKUs
        if "combined" in stem.lower():
            return None

        # Build a clean descriptor: drop brand/state code prefix and asset-type/version words
        descriptor = stem
        # Strip common prefix patterns like "HH_NY_" or "HH_CA_"
        descriptor = re.sub(r"^HH[_ ]?[A-Z]{2}[_ ]?", "", descriptor, flags=re.IGNORECASE)
        # Strip asset-type and trailing version words
        strip_words = [
            "sell sheet", "sellsheet", "postcard", "post card",
            "tabletent", "table tent", "tradeshow", "trade show",
            "partner guide", "partnerguide", "training deck", "launch deck",
            "withcrops", "with crops", "crops", "final", "draft",
        ]
        for w in strip_words:
            descriptor = re.sub(re.escape(w), " ", descriptor, flags=re.IGNORECASE)
        # Strip version numbers like "_1.2", "_v3", "_1.1"
        descriptor = re.sub(r"[_ ]+v?\d+(\.\d+)*", " ", descriptor)
        # Replace separators with spaces
        descriptor = re.sub(r"[_\-&]+", " ", descriptor)
        descriptor = re.sub(r"\s+", " ", descriptor).strip()

        if len(descriptor) < 3:
            return None

        verdict = self._call_matcher(
            brand=brand,
            candidate=descriptor,
            context=f"DOCUMENT: {filename}",
            content_type="file_level",
            primary_product=primary_product,
        )
        if verdict.get("decision") == "matched":
            logger.info(f"file-level SKU resolved: {filename!r} -> {verdict.get('product_name')!r}")
            return verdict
        return None

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

        # File-level SKU resolution: single-SKU files get matched once, applied to all blocks
        file_verdict = self.resolve_file_level_sku(filename, asset_type or "", brand, product or "")

        total_saved = 0
        for page_data in pages:
            blocks = self.extract_all_content(page_data, brand)
            if not blocks:
                continue

            if file_verdict:
                # Stamp the file-level SKU onto every block, skip per-block matching
                for block in blocks:
                    block["_matcher_decision"] = "matched"
                    block["product_name"] = file_verdict.get("product_line") or file_verdict.get("product_name")
                    block["_matched_inventory_id"] = file_verdict.get("inventory_id")
                    block["_matched_subcategory"] = file_verdict.get("subcategory")
                    block["_matched_category"] = file_verdict.get("category")
            else:
                # Fall back to per-block matching for multi-SKU files
                blocks = self.match_blocks(
                    blocks, brand, page_data.get("text", ""),
                    filename=filename, primary_product=product or "",
                )

            if not blocks:
                continue
            total_saved += self.save_to_silo(
                silo_table=silo_table, blocks=blocks, page_data=page_data,
                brand=brand, state=state, category=category, product=product,
                asset_type=asset_type, source_path=source_path, state_list=state_list,
            )
            time.sleep(0.3)

        return {"brand": brand, "state": state, "silo_table": silo_table,
                "pages": len(pages), "blocks_saved": total_saved,
                "file_level_match": file_verdict.get("inventory_id") if file_verdict else None}
