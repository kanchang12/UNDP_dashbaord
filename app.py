# MIT Licence — CrisisMap, UNDP Crisis Damage Reporting
# Flask backend.
#
# Required env vars:
#   SUPABASE_URL        — e.g. https://xxxx.supabase.co
#   SUPABASE_ANON_KEY   — Supabase anon or service_role key
#   GEMINI_API_KEY      — Google Gemini API key
#   OPENAI_API_KEY      — OpenAI API key (optional, GPT-4o fallback)

import base64
import json
import logging
import math
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Union

import requests
from flask import Flask, request, Response, render_template

app = Flask(__name__, template_folder="templates", static_folder="static")

logging.basicConfig(level=logging.INFO)

# ── Environment variables ──────────────────────────────────────────────────────
from google import genai
from google.genai import types

SUPABASE_URL      = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
from google import genai
from google.genai import types
client = genai.Client(api_key=GEMINI_API_KEY)
STORAGE_BUCKET = "report-images"

_rate_limit_store: Dict[str, List[datetime]] = defaultdict(list)
RATE_LIMIT_MAX    = 10
RATE_LIMIT_WINDOW = 60

# ── CORS ───────────────────────────────────────────────────────────────────────

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, PATCH, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


def cors_response(body: str, status: int = 200, mimetype: str = "application/json") -> Response:
    return Response(body, status=status, mimetype=mimetype, headers=CORS_HEADERS)


def options_response() -> Response:
    return Response("", status=204, headers=CORS_HEADERS)


def is_rate_limited(device_id: str) -> bool:
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW)
    timestamps = _rate_limit_store[device_id]
    timestamps[:] = [t for t in timestamps if t > window_start]
    if len(timestamps) >= RATE_LIMIT_MAX:
        return True
    timestamps.append(now)
    return False


def get_body():
    """Accept JSON from both GET (query string) and POST (body)."""
    if request.method == "POST":
        return request.get_json(silent=True) or {}
    else:
        return dict(request.args)


# ── Root ───────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# ── Supabase helpers ───────────────────────────────────────────────────────────

def _sb_headers(content_type: str = "application/json") -> dict:
    return {
        "apikey": SUPABASE_ANON_KEY,
        "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
        "Content-Type": content_type,
        "Prefer": "return=representation",
    }


def sb_insert(table: str, data: dict) -> Optional[dict]:
    try:
        r = requests.post(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=_sb_headers(),
            json=data,
            timeout=10,
        )
        r.raise_for_status()
        rows = r.json()
        return rows[0] if rows else data
    except Exception as e:
        logging.error(f"sb_insert {table}: {e}")
        return None


def sb_select(table: str, params: Optional[dict] = None, single: bool = False) -> Union[List, Optional[dict]]:
    try:
        r = requests.get(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers={**_sb_headers(), "Accept": "application/json"},
            params=params or {},
            timeout=10,
        )
        r.raise_for_status()
        data = r.json()
        return data[0] if single and data else data
    except Exception as e:
        logging.error(f"sb_select {table}: {e}")
        return None if single else []


def sb_update(table: str, match: dict, data: dict) -> None:
    try:
        params = {k: f"eq.{v}" for k, v in match.items()}
        requests.patch(
            f"{SUPABASE_URL}/rest/v1/{table}",
            headers=_sb_headers(),
            params=params,
            json=data,
            timeout=10,
        )
    except Exception as e:
        logging.error(f"sb_update {table}: {e}")


# ── Storage ────────────────────────────────────────────────────────────────────

def _detect_mime(image_b64: str) -> str:
    try:
        header = base64.b64decode(image_b64[:16])
        if header[:8] == b'\x89PNG\r\n\x1a\n':
            return "image/png"
        if header[:6] in (b'GIF87a', b'GIF89a'):
            return "image/gif"
        if header[:4] == b'RIFF' and header[8:12] == b'WEBP':
            return "image/webp"
    except Exception:
        pass
    return "image/jpeg"


def _mime_to_ext(mime: str) -> str:
    return {"image/png": "png", "image/gif": "gif", "image/webp": "webp"}.get(mime, "jpg")


def upload_image_to_storage(image_b64: str, report_id: str, index: int) -> Optional[str]:
    try:
        image_bytes = base64.b64decode(image_b64)
        mime = _detect_mime(image_b64)
        ext = _mime_to_ext(mime)
        filename = f"{report_id}/{index}.{ext}"
        r = requests.post(
            f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{filename}",
            headers={
                "apikey": SUPABASE_ANON_KEY,
                "Authorization": f"Bearer {SUPABASE_ANON_KEY}",
                "Content-Type": mime,
            },
            data=image_bytes,
            timeout=20,
        )
        if r.status_code in (200, 201):
            return f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{filename}"
        logging.warning(f"Storage upload {index}: {r.status_code} {r.text[:120]}")
        return None
    except Exception as e:
        logging.error(f"upload_image_to_storage: {e}")
        return None


# ── Haversine ──────────────────────────────────────────────────────────────────

def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── Gemini helpers ─────────────────────────────────────────────────────────────

def _gemini_post(payload: dict, timeout: int = 30) -> Optional[dict]:
    try:
        r = requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json=payload,
            timeout=timeout,
        )
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logging.error(f"Gemini call failed: {e}")
        return None


def _extract_json_from_gemini(response: dict) -> Optional[dict]:
    try:
        text = response["candidates"][0]["content"]["parts"][0]["text"]
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception as e:
        logging.warning(f"JSON extraction failed: {e}")
    return None


# ── STEP 1 — AI reads images and fills ALL UNDP fields ────────────────────────

def step1_image_analysis(image_b64_list: List[str]) -> dict:
    if not GEMINI_API_KEY or not image_b64_list:
        return {"damage_level": "minimal", "infrastructure_type": "other", "infrastructure_name": "", "crisis_type": "insignificant", "has_debris": "attention_not_required", "debris_detected": False, "electricity_status": "attention_not_required", "health_services_status": "attention_not_required", "pressing_needs": [], "location_description": "No images.", "confidence": 0.0, "reasoning": "No images."}
    
    prompt = """Analyze these images. Return ONLY JSON:
    {
      "damage_level": "minimal | partial | complete",
      "infrastructure_type": "residential | commercial | government | utility | transport | community | public | other",
      "infrastructure_name": "string",
      "crisis_type": "earthquake | flood | tsunami | hurricane | wildfire | explosion | chemical | conflict | civil_unrest | insignificant",
      "has_debris": "yes | no | attention_not_required",
      "debris_detected": bool,
      "electricity_status": "no_damage | minor_damage | moderate_damage | severe_damage | destroyed | attention_not_required",
      "health_services_status": "fully_functional | partially_functional | largely_disrupted | not_functioning | attention_not_required",
      "pressing_needs": ["food_water", "cash_assistance", "healthcare", "shelter", "livelihoods", "wash", "infrastructure_restoration", "psychosocial_support", "authority_support", "other"],
      "location_description": "string",
      "confidence": 0.0,
      "reasoning": "string"
    }"""
    
    parts = [prompt]
    for b64 in image_b64_list[:4]:
        try:
            parts.append(types.Part.from_bytes(data=base64.b64decode(b64), mime_type=_detect_mime(b64)))
        except: continue
        
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash",
            contents=parts,
            config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.1)
        )
        result = json.loads(response.text)
        if not isinstance(result.get("pressing_needs"), list): result["pressing_needs"] = []
        return result
    except Exception as e:
        logging.error(f"Gemini failure: {e}")
        return {"damage_level": "minimal", "infrastructure_type": "other", "reasoning": "AI error."}


# ── STEP 2 — Area signals ──────────────────────────────────────────────────────

def step2_area_signals(lat: Optional[float], lng: Optional[float]) -> dict:
    if lat is None or lng is None:
        return {"nearby_count": 0, "avg_damage": None, "dominant_crisis": None, "density_per_km2": 0}

    two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    lat_delta = 0.0045
    lng_delta = lat_delta / max(math.cos(math.radians(lat)), 0.01)

    rows = sb_select("reports", params={
        "and": (
            f"(lat.gte.{lat - lat_delta},"
            f"lat.lte.{lat + lat_delta},"
            f"lng.gte.{lng - lng_delta},"
            f"lng.lte.{lng + lng_delta},"
            f"submitted_at.gte.{two_hours_ago})"
        ),
        "select": "lat,lng,damage_level,crisis_type",
        "limit": "200",
    }) or []

    nearby = [
        r for r in rows
        if r.get("lat") and r.get("lng")
        and haversine_m(lat, lng, float(r["lat"]), float(r["lng"])) <= 500
    ]

    if not nearby:
        return {"nearby_count": 0, "avg_damage": None, "dominant_crisis": None, "density_per_km2": 0}

    damage_scores = {"minimal": 1, "partial": 2, "complete": 3}
    levels = [damage_scores.get(r.get("damage_level", ""), 0) for r in nearby if r.get("damage_level")]
    avg_score = sum(levels) / len(levels) if levels else 0

    crisis_counts: Dict[str, int] = {}
    for r in nearby:
        ct = r.get("crisis_type", "")
        if ct:
            crisis_counts[ct] = crisis_counts.get(ct, 0) + 1
    dominant = max(crisis_counts, key=lambda k: crisis_counts[k]) if crisis_counts else None

    avg_damage = (
        "minimal" if avg_score < 1.5
        else "partial" if avg_score < 2.5
        else "complete"
    )

    return {
        "nearby_count": len(nearby),
        "avg_damage": avg_damage,
        "dominant_crisis": dominant,
        "density_per_km2": round(len(nearby) / (math.pi * 0.5 ** 2), 1),
    }


# ── STEP 3 — Combined assessment ───────────────────────────────────────────────

def step3_combined_assessment(image_analysis: dict, area_signals: dict) -> dict:
    system_prompt = (
        "You are a UNDP crisis assessment AI. "
        "Given image analysis and surrounding area signals, "
        "produce a final priority score and recommended action for emergency responders. "
        "Return JSON only, no markdown."
    )
    user_content = (
        f"Image analysis: {json.dumps(image_analysis)}\n"
        f"Area signals (500m radius, last 2h): {json.dumps(area_signals)}\n\n"
        'Return JSON: {"final_damage_level": "minimal|partial|complete", '
        '"priority_score": 0-100, "confidence": 0.0-1.0, '
        '"recommended_action": "one sentence for responders", "reasoning": "two sentences max"}'
    )

    if OPENAI_API_KEY:
        try:
            r = requests.post(
                OPENAI_URL,
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
                json={
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_content},
                    ],
                    "max_tokens": 300,
                    "temperature": 0.1,
                },
                timeout=20,
            )
            r.raise_for_status()
            text = r.json()["choices"][0]["message"]["content"]
            start, end = text.find("{"), text.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(text[start:end])
        except Exception as e:
            logging.warning(f"OpenAI step3 failed, falling back to Gemini: {e}")

    payload = {"contents": [{"parts": [{"text": f"System: {system_prompt}\n\nUser: {user_content}"}]}]}
    response = _gemini_post(payload, timeout=20)
    if response:
        result = _extract_json_from_gemini(response)
        if result:
            return result

    final = image_analysis.get("damage_level", "partial")
    priority = {"minimal": 20, "partial": 55, "complete": 85}.get(final, 30)
    return {
        "final_damage_level": final,
        "priority_score": priority,
        "confidence": image_analysis.get("confidence", 0.3),
        "recommended_action": "Dispatch assessment team to verify damage.",
        "reasoning": "Automated combined assessment unavailable; using image analysis values.",
    }


# ── STEP 4 — Duplicate detection ───────────────────────────────────────────────

def step4_duplicate_detection(
    building_footprint_id: Optional[str],
    device_id: str,
    report_id: str,
) -> Tuple[bool, Optional[str]]:
    if not building_footprint_id:
        return False, None

    existing = sb_select("reports", params={
        "building_footprint_id": f"eq.{building_footprint_id}",
        "id": f"neq.{report_id}",
        "select": "id,version_group_id,anonymous_device_id,submitted_at",
        "order": "submitted_at.desc",
        "limit": "10",
    }) or []

    if not existing:
        return False, None

    version_group_id = None
    for row in existing:
        if row.get("version_group_id"):
            version_group_id = row["version_group_id"]
            break
    if not version_group_id:
        version_group_id = str(uuid.uuid4())

    is_flagged = False
    ten_min_ago = datetime.now(timezone.utc) - timedelta(minutes=10)
    for row in existing:
        if row.get("anonymous_device_id") == device_id:
            try:
                ts = datetime.fromisoformat(row["submitted_at"].replace("Z", "+00:00"))
                if ts > ten_min_ago:
                    is_flagged = True
                    break
            except Exception:
                pass

    return is_flagged, version_group_id


def compute_area_signal_score(area_signals: dict, combined: dict) -> float:
    score = combined.get("priority_score", 30) / 100
    density_boost = min(area_signals.get("density_per_km2", 0) / 20, 0.3)
    return round(min(score + density_boost, 1.0), 3)


# ── ENDPOINTS ──────────────────────────────────────────────────────────────────

@app.route("/api/submit-report", methods=["GET", "POST", "OPTIONS"])
def submit_report():
    if request.method == "OPTIONS":
        return options_response()

    body = get_body()

    device_id = body.get("anonymous_device_id", "")
    if not device_id:
        return cors_response(json.dumps({"error": "anonymous_device_id required"}), 400)

    if is_rate_limited(device_id):
        return cors_response(json.dumps({"error": "Rate limit exceeded."}), 429)

    report_id    = str(uuid.uuid4())
    submitted_at = datetime.now(timezone.utc).isoformat()
    lat          = body.get("lat")
    lng          = body.get("lng")

    images_b64: List[str] = body.get("images_b64", [])
    if isinstance(images_b64, str):
        images_b64 = [images_b64]

    image_urls: List[str] = []
    for i, b64 in enumerate(images_b64[:4]):
        url = upload_image_to_storage(b64, report_id, i)
        if url:
            image_urls.append(url)

    if not image_urls:
        image_urls = body.get("image_urls", [])
        if isinstance(image_urls, str):
            image_urls = [image_urls]

    image_analysis = step1_image_analysis(images_b64[:4] if images_b64 else [])
    area_signals   = step2_area_signals(
        float(lat) if lat is not None else None,
        float(lng) if lng is not None else None,
    )
    combined       = step3_combined_assessment(image_analysis, area_signals)
    is_dup, vg_id  = step4_duplicate_detection(body.get("building_footprint_id"), device_id, report_id)
    area_score     = compute_area_signal_score(area_signals, combined)

    pressing_needs = image_analysis.get("pressing_needs", [])
    if isinstance(pressing_needs, str):
        pressing_needs = [pressing_needs]

    record = {
        "id":                        report_id,
        "submitted_at":              submitted_at,
        "lat":                       float(lat) if lat is not None else None,
        "lng":                       float(lng) if lng is not None else None,
        "building_footprint_id":     body.get("building_footprint_id"),
        "location_description":      image_analysis.get("location_description", ""),
        "infrastructure_type":       image_analysis.get("infrastructure_type", "other"),
        "infrastructure_type_other": "",
        "infrastructure_name":       image_analysis.get("infrastructure_name", ""),
        "damage_level":              image_analysis.get("damage_level", "partial"),
        "crisis_type":               image_analysis.get("crisis_type", "unknown"),
        "has_debris":                image_analysis.get("has_debris", "unknown"),
        "electricity_status":        image_analysis.get("electricity_status", "unknown"),
        "health_services_status":    image_analysis.get("health_services_status", "unknown"),
        "pressing_needs":            pressing_needs,
        "image_urls":                image_urls,
        "ai_damage_classification":  combined.get("final_damage_level"),
        "ai_confidence_score":       combined.get("confidence"),
        "ai_reasoning":              combined.get("reasoning"),
        "ai_recommended_action":     combined.get("recommended_action"),
        "ai_priority_score":         combined.get("priority_score"),
        "area_signal_score":         area_score,
        "is_duplicate_flagged":      is_dup,
        "version_group_id":          vg_id,
        "language_submitted":        body.get("language_submitted", "en"),
        "anonymous_device_id":       device_id,
        "user_confirmed":            False,
    }

    sb_insert("reports", record)

    if vg_id and is_dup and body.get("building_footprint_id"):
        sb_update("reports", {"building_footprint_id": body.get("building_footprint_id")}, {"version_group_id": vg_id})

    return cors_response(json.dumps({
        "submission_id":         report_id,
        "image_urls":            image_urls,
        "is_duplicate_flagged":  is_dup,
        "area_signals":          area_signals,
        "ai_filled": {
            "damage_level":           image_analysis.get("damage_level"),
            "infrastructure_type":    image_analysis.get("infrastructure_type"),
            "infrastructure_name":    image_analysis.get("infrastructure_name"),
            "crisis_type":            image_analysis.get("crisis_type"),
            "has_debris":             image_analysis.get("has_debris"),
            "electricity_status":     image_analysis.get("electricity_status"),
            "health_services_status": image_analysis.get("health_services_status"),
            "pressing_needs":         image_analysis.get("pressing_needs"),
            "location_description":   image_analysis.get("location_description"),
            "confidence":             image_analysis.get("confidence"),
            "reasoning":              image_analysis.get("reasoning"),
        },
        "combined_assessment": {
            "final_damage_level":  combined.get("final_damage_level"),
            "priority_score":      combined.get("priority_score"),
            "recommended_action":  combined.get("recommended_action"),
            "reasoning":           combined.get("reasoning"),
        },
    }), 201)


@app.route("/api/confirm-report", methods=["GET", "POST", "PATCH", "OPTIONS"])
def confirm_report():
    if request.method == "OPTIONS":
        return options_response()

    body = get_body()
    if not body or not body.get("submission_id"):
        return cors_response(json.dumps({"error": "submission_id required"}), 400)

    update_data = {"user_confirmed": True}
    allowed = [
        "damage_level", "infrastructure_type", "infrastructure_name",
        "crisis_type", "has_debris", "electricity_status",
        "health_services_status", "pressing_needs", "location_description",
    ]
    corrections = body.get("corrections", {})
    for field in allowed:
        if field in corrections:
            update_data[field] = corrections[field]

    sb_update("reports", {"id": body["submission_id"]}, update_data)
    return cors_response(json.dumps({"status": "confirmed", "submission_id": body["submission_id"]}), 200)


@app.route("/api/reports", methods=["GET", "POST", "OPTIONS"])
def get_reports():
    if request.method == "OPTIONS":
        return options_response()

    p = request.args if request.method == "GET" else (request.get_json(silent=True) or request.args)
    filters = []

    if p.get("crisis_type"):
        filters.append(f"crisis_type.eq.{p['crisis_type']}")
    if p.get("damage_level"):
        filters.append(f"damage_level.eq.{p['damage_level']}")
    if p.get("infrastructure_type"):
        filters.append(f"infrastructure_type.eq.{p['infrastructure_type']}")
    if p.get("date_from"):
        filters.append(f"submitted_at.gte.{p['date_from']}")
    if p.get("date_to"):
        filters.append(f"submitted_at.lte.{p['date_to']}")

    params: dict = {"order": "submitted_at.desc", "limit": "2000"}
    if filters:
        params["and"] = f"({','.join(filters)})"

    rows = sb_select("reports", params=params) or []

    bbox = p.get("bbox")
    if bbox:
        try:
            min_lng, min_lat, max_lng, max_lat = map(float, bbox.split(","))
            rows = [
                r for r in rows
                if r.get("lat") and r.get("lng")
                and min_lat <= float(r["lat"]) <= max_lat
                and min_lng <= float(r["lng"]) <= max_lng
            ]
        except Exception:
            pass

    fmt = p.get("format", "geojson")
    if fmt == "csv":
        return _rows_to_csv_response(rows)

    return cors_response(json.dumps(_rows_to_geojson(rows)))


@app.route("/api/dashboard-stats", methods=["GET", "POST", "OPTIONS"])
def dashboard_stats():
    if request.method == "OPTIONS":
        return options_response()

    rows = sb_select("reports", params={"order": "submitted_at.desc", "limit": "5000"}) or []

    total         = len(rows)
    damage_counts = _count_by(rows, "damage_level")
    infra_counts  = _count_by(rows, "infrastructure_type")
    crisis_counts = _count_by(rows, "crisis_type")

    now = datetime.now(timezone.utc)
    buckets: Dict[str, int] = {}
    for h in range(24):
        label = (now - timedelta(hours=23 - h)).strftime("%H:00")
        buckets[label] = 0
    for r in rows:
        ts_str = r.get("submitted_at", "")
        try:
            ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            if ts >= now - timedelta(hours=24):
                label = ts.strftime("%H:00")
                if label in buckets:
                    buckets[label] += 1
        except Exception:
            pass

    intensity_map = {"minimal": 0.3, "partial": 0.6, "complete": 1.0}
    heatmap = [
        [float(r["lat"]), float(r["lng"]), intensity_map.get(r.get("damage_level", ""), 0.5)]
        for r in rows if r.get("lat") and r.get("lng")
    ]

    priority_rows = sorted(
        [r for r in rows if r.get("ai_priority_score") is not None],
        key=lambda r: float(r["ai_priority_score"]),
        reverse=True,
    )[:10]
    priority_queue = [
        {
            "id":                    r.get("id"),
            "lat":                   r.get("lat"),
            "lng":                   r.get("lng"),
            "infrastructure_type":   r.get("infrastructure_type"),
            "damage_level":          r.get("damage_level"),
            "ai_priority_score":     r.get("ai_priority_score"),
            "ai_recommended_action": r.get("ai_recommended_action"),
            "submitted_at":          r.get("submitted_at"),
        }
        for r in priority_rows
    ]

    return cors_response(json.dumps({
        "total":          total,
        "damage_counts":  damage_counts,
        "infra_counts":   infra_counts,
        "crisis_counts":  crisis_counts,
        "timeline_24h":   [{"hour": k, "count": v} for k, v in buckets.items()],
        "heatmap_points": heatmap,
        "priority_queue": priority_queue,
        "last_updated":   now.isoformat(),
    }))


@app.route("/api/export", methods=["GET", "POST", "OPTIONS"])
def export_data():
    if request.method == "OPTIONS":
        return options_response()

    body       = get_body()
    filters_in = body.get("filters", {})
    if isinstance(filters_in, str):
        filters_in = {}
    fmt = body.get("format", "csv")

    filters = []
    if filters_in.get("crisis_type"):
        filters.append(f"crisis_type.eq.{filters_in['crisis_type']}")
    if filters_in.get("damage_level"):
        filters.append(f"damage_level.eq.{filters_in['damage_level']}")
    if filters_in.get("infrastructure_type"):
        filters.append(f"infrastructure_type.eq.{filters_in['infrastructure_type']}")
    if filters_in.get("date_from"):
        filters.append(f"submitted_at.gte.{filters_in['date_from']}")
    if filters_in.get("date_to"):
        filters.append(f"submitted_at.lte.{filters_in['date_to']}")

    params: dict = {"order": "submitted_at.desc", "limit": "10000"}
    if filters:
        params["and"] = f"({','.join(filters)})"

    rows = sb_select("reports", params=params) or []

    if fmt == "geojson":
        return Response(
            json.dumps(_rows_to_geojson(rows), indent=2),
            status=200,
            mimetype="application/geo+json",
            headers={
                **CORS_HEADERS,
                "Content-Disposition": "attachment; filename=crisismap_export.geojson",
            },
        )

    return _rows_to_csv_response(rows)


@app.route("/api/reporter-alert", methods=["GET", "POST", "OPTIONS"])
def reporter_alert():
    if request.method == "OPTIONS":
        return options_response()

    body = get_body()
    device_id = body.get("anonymous_device_id", "unknown")
    lat = body.get("lat")
    lng = body.get("lng")
    report_id = str(uuid.uuid4())
    now_iso = datetime.now(timezone.utc).isoformat()

    # 1. Always store the safety alert in the secondary table
    alert_record = {
        "id": str(uuid.uuid4()),
        "alert_type": body.get("alert_type", "REPORTER_SAFETY_ALERT"),
        "anonymous_device_id": device_id,
        "coordinator_id": body.get("coordinator_id", ""),
        "lat": lat,
        "lng": lng,
        "accuracy_metres": body.get("accuracy_metres"),
        "triggered_at": body.get("triggered_at") or now_iso,
    }
    sb_insert("reporter_alerts", alert_record)
    logging.info(f"Alert stored for {device_id}")

    # 2. Extract and sanitize images
    images_b64 = body.get("images_b64", [])
    if isinstance(images_b64, str):
        images_b64 = [images_b64]

    # 3. Initialize default values (In case images are missing or AI fails)
    image_urls = []
    image_analysis = {
        "damage_level": "minimal",
        "infrastructure_type": "other",
        "crisis_type": "insignificant",
        "location_description": "Alert triggered via reporter safety",
        "pressing_needs": [],
        "has_debris": "attention_not_required",
        "electricity_status": "attention_not_required",
        "health_services_status": "attention_not_required",
        "infrastructure_name": "Unknown (Alert Only)"
    }
    combined = {
        "final_damage_level": "minimal",
        "priority_score": 50,
        "confidence": 0.5,
        "recommended_action": "Immediate safety alert: Manual verification required.",
        "reasoning": "Report generated via emergency reporter-alert trigger."
    }
    area_score = 0.5

    # 4. Run full AI pipeline ONLY if images exist
    if images_b64:
        # Upload images
        for i, b64 in enumerate(images_b64[:4]):
            url = upload_image_to_storage(b64, report_id, i)
            if url: image_urls.append(url)

        # AI Analysis
        image_analysis = step1_image_analysis(images_b64[:4])
        area_signals = step2_area_signals(float(lat) if lat else None, float(lng) if lng else None)
        combined = step3_combined_assessment(image_analysis, area_signals)
        area_score = compute_area_signal_score(area_signals, combined)

    # 5. Determine duplication and grouping
    is_dup, vg_id = step4_duplicate_detection(body.get("building_footprint_id"), device_id, report_id)

    # 6. FULL REPORT RECORD - This MUST be outside any 'if' to update the map
    report_record = {
        "id": report_id,
        "submitted_at": now_iso,
        "lat": float(lat) if lat is not None else None,
        "lng": float(lng) if lng is not None else None,
        "building_footprint_id": body.get("building_footprint_id"),
        "location_description": image_analysis.get("location_description", ""),
        "infrastructure_type": image_analysis.get("infrastructure_type", "other"),
        "infrastructure_type_other": "",
        "infrastructure_name": image_analysis.get("infrastructure_name", ""),
        "damage_level": image_analysis.get("damage_level", "minimal"),
        "crisis_type": image_analysis.get("crisis_type", "insignificant"),
        "has_debris": image_analysis.get("has_debris", "attention_not_required"),
        "electricity_status": image_analysis.get("electricity_status", "attention_not_required"),
        "health_services_status": image_analysis.get("health_services_status", "attention_not_required"),
        "pressing_needs": image_analysis.get("pressing_needs", []),
        "image_urls": image_urls,
        "ai_damage_classification": combined.get("final_damage_level"),
        "ai_confidence_score": combined.get("confidence"),
        "ai_reasoning": combined.get("reasoning"),
        "ai_recommended_action": combined.get("recommended_action"),
        "ai_priority_score": combined.get("priority_score"),
        "area_signal_score": area_score,
        "is_duplicate_flagged": is_dup,
        "version_group_id": vg_id,
        "language_submitted": body.get("language_submitted", "en"),
        "anonymous_device_id": device_id,
        "user_confirmed": False,
    }

    # Final Save to Dashboard Table
    sb_insert("reports", report_record)
    logging.info(f"Dashboard updated for report: {report_id}")

    return cors_response(json.dumps({
        "status": "received",
        "report_created": True,
        "submission_id": report_id,
        "ai_filled": image_analysis,
        "combined_assessment": combined
    }), 201)


# ── Serialisation ──────────────────────────────────────────────────────────────

def _rows_to_geojson(rows: list) -> dict:
    features = []
    for r in rows:
        if not r.get("lat") or not r.get("lng"):
            continue
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(r["lng"]), float(r["lat"])]},
            "properties": {
                "id":                       r.get("id"),
                "submitted_at":             r.get("submitted_at"),
                "infrastructure_type":      r.get("infrastructure_type"),
                "infrastructure_name":      r.get("infrastructure_name"),
                "damage_level":             r.get("damage_level"),
                "crisis_type":              r.get("crisis_type"),
                "has_debris":               r.get("has_debris"),
                "electricity_status":       r.get("electricity_status"),
                "health_services_status":   r.get("health_services_status"),
                "pressing_needs":           r.get("pressing_needs", []),
                "location_description":     r.get("location_description"),
                "building_footprint_id":    r.get("building_footprint_id"),
                "image_urls":               r.get("image_urls", []),
                "ai_damage_classification": r.get("ai_damage_classification"),
                "ai_confidence_score":      r.get("ai_confidence_score"),
                "ai_reasoning":             r.get("ai_reasoning"),
                "ai_recommended_action":    r.get("ai_recommended_action"),
                "ai_priority_score":        r.get("ai_priority_score"),
                "area_signal_score":        r.get("area_signal_score"),
                "is_duplicate_flagged":     r.get("is_duplicate_flagged"),
                "version_group_id":         r.get("version_group_id"),
                "language_submitted":       r.get("language_submitted"),
                "user_confirmed":           r.get("user_confirmed"),
            },
        })
    return {"type": "FeatureCollection", "features": features}


_CSV_FIELDS = [
    "id", "submitted_at", "lat", "lng", "building_footprint_id",
    "location_description", "infrastructure_type", "infrastructure_type_other",
    "infrastructure_name", "damage_level", "crisis_type", "has_debris",
    "electricity_status", "health_services_status", "pressing_needs",
    "image_urls", "ai_damage_classification", "ai_confidence_score",
    "ai_reasoning", "ai_recommended_action", "ai_priority_score",
    "area_signal_score", "is_duplicate_flagged", "version_group_id",
    "language_submitted", "anonymous_device_id", "user_confirmed",
]


def _rows_to_csv_response(rows: list) -> Response:
    lines = [",".join(_CSV_FIELDS)]
    for r in rows:
        def _cell(v):
            if isinstance(v, list):
                v = "|".join(str(i) for i in v)
            if v is None:
                return ""
            s = str(v).replace('"', '""')
            return f'"{s}"' if ("," in s or "\n" in s or '"' in s) else s
        lines.append(",".join(_cell(r.get(f)) for f in _CSV_FIELDS))

    csv_bytes = ("\ufeff" + "\n".join(lines)).encode("utf-8")
    return Response(
        csv_bytes,
        status=200,
        mimetype="text/csv; charset=utf-8",
        headers={
            **CORS_HEADERS,
            "Content-Disposition": "attachment; filename=crisismap_export.csv",
        },
    )


def _count_by(rows: list, field: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for r in rows:
        val = r.get(field, "unknown") or "unknown"
        counts[val] = counts.get(val, 0) + 1
    return counts


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
