# MIT Licence — CrisisMap, UNDP Crisis Damage Reporting
# Flask backend. Set environment variables in a .env file or your hosting
# platform's config — never hardcoded.
#
# Required env vars:
#   SUPABASE_URL        — e.g. https://xxxx.supabase.co
#   SUPABASE_ANON_KEY   — Supabase anon or service_role key
#   GEMINI_API_KEY      — Google Gemini API key (image analysis + Step 3 fallback)
#   OPENAI_API_KEY      — OpenAI API key (Step 3 combined assessment, GPT-4o)
#
# Run locally:
#   pip install -r requirements.txt
#   python app.py

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

# FIX 1 — explicit template and static folders so render_template works
app = Flask(__name__, template_folder="templates", static_folder="static")

logging.basicConfig(level=logging.INFO)

# ── Environment variables ──────────────────────────────────────────────────────

SUPABASE_URL      = os.environ.get("SUPABASE_URL", "")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "")
GEMINI_API_KEY    = os.environ.get("GEMINI_API_KEY", "")
OPENAI_API_KEY    = os.environ.get("OPENAI_API_KEY", "")

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/"
    "models/gemini-1.5-flash:generateContent"
)
STORAGE_BUCKET = "report-images"

# FIX 7 — simple in-memory rate limit store (device_id -> list of timestamps)
_rate_limit_store: Dict[str, List[datetime]] = defaultdict(list)
RATE_LIMIT_MAX    = 10   # max submissions
RATE_LIMIT_WINDOW = 60   # per N seconds


# ── CORS helpers ───────────────────────────────────────────────────────────────

CORS_HEADERS = {
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type, Authorization",
}


def cors_response(body: str, status: int = 200, mimetype: str = "application/json") -> Response:
    return Response(body, status=status, mimetype=mimetype, headers=CORS_HEADERS)


def options_response() -> Response:
    return Response("", status=204, headers=CORS_HEADERS)


def is_rate_limited(device_id: str) -> bool:
    """FIX 7 — reject if device submits more than RATE_LIMIT_MAX times in RATE_LIMIT_WINDOW seconds."""
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(seconds=RATE_LIMIT_WINDOW)
    timestamps = _rate_limit_store[device_id]
    timestamps[:] = [t for t in timestamps if t > window_start]
    if len(timestamps) >= RATE_LIMIT_MAX:
        return True
    timestamps.append(now)
    return False


# ── Root route — serves dashboard ─────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


# ── Supabase REST helpers ──────────────────────────────────────────────────────

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


# ── Supabase Storage ───────────────────────────────────────────────────────────

def _detect_mime(image_b64: str) -> str:
    """FIX 5 — detect actual image mime type from magic bytes, not assume JPEG."""
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


# ── Haversine distance (metres) ────────────────────────────────────────────────

def haversine_m(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ── AI Pipeline ────────────────────────────────────────────────────────────────

def _gemini_post(payload: dict, timeout: int = 20) -> Optional[dict]:
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


def step1_image_analysis(image_b64_list: List[str]) -> dict:
    """
    Step 1 — Gemini Vision API.
    Analyses all submitted images together and returns a structured damage assessment.
    """
    if not GEMINI_API_KEY or not image_b64_list:
        return {
            "damage_level": "unknown",
            "infrastructure_type": "unknown",
            "debris_detected": False,
            "confidence": 0.0,
            "reasoning": "Image analysis unavailable — no API key or images.",
        }

    parts = [
        {
            "text": (
                "Analyse these images of infrastructure damage. "
                "Classify damage as exactly one of: minimal, partial, or complete. "
                "Identify the infrastructure type visible. "
                "Detect any debris present. "
                "Provide a confidence score 0-1. "
                "Return JSON only, no markdown: "
                '{"damage_level": "minimal|partial|complete", '
                '"infrastructure_type": "residential|commercial|government|utility|transport|community|public|other", '
                '"debris_detected": true|false, '
                '"confidence": 0.0-1.0, '
                '"reasoning": "one sentence"}'
            )
        }
    ]

    for b64 in image_b64_list[:4]:
        mime = _detect_mime(b64)
        parts.append({
            "inline_data": {
                "mime_type": mime,
                "data": b64,
            }
        })

    payload = {"contents": [{"parts": parts}]}
    response = _gemini_post(payload, timeout=30)
    if response:
        result = _extract_json_from_gemini(response)
        if result:
            return result

    return {
        "damage_level": "unknown",
        "infrastructure_type": "unknown",
        "debris_detected": False,
        "confidence": 0.0,
        "reasoning": "Gemini Vision analysis returned no parseable result.",
    }


def step2_area_signals(lat: Optional[float], lng: Optional[float]) -> dict:
    """
    Step 2 — Area signal aggregation.
    FIX 4 — queries BOTH sides of the bounding box (min AND max lat/lng).
    """
    if lat is None or lng is None:
        return {"nearby_count": 0, "avg_damage": None, "dominant_crisis": None, "density_per_km2": 0}

    two_hours_ago = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    lat_delta = 0.0045  # ~500m
    lng_delta = lat_delta / max(math.cos(math.radians(lat)), 0.01)

    # FIX 4 — use PostgREST and() to apply both gte and lte on same columns
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


def step3_combined_assessment(
    image_analysis: dict,
    area_signals: dict,
    form_data: dict,
) -> dict:
    """
    Step 3 — Combined AI decision.
    Sends image analysis + area signals + form data to OpenAI GPT-4o,
    with Gemini as fallback.
    """
    system_prompt = (
        "You are a UNDP crisis assessment AI. "
        "Given image analysis results and surrounding area signals, "
        "make a final damage assessment and priority score for emergency responders. "
        "Be objective and data-driven. Return JSON only, no markdown."
    )
    user_content = (
        f"Image analysis: {json.dumps(image_analysis)}\n"
        f"Area signals (500m radius, last 2h): {json.dumps(area_signals)}\n"
        f"Reported infrastructure type: {form_data.get('infrastructure_type', 'unknown')}\n"
        f"Reported damage level: {form_data.get('damage_level', 'unknown')}\n"
        f"Crisis type: {form_data.get('crisis_type', 'unknown')}\n"
        f"Debris reported: {form_data.get('has_debris', 'unknown')}\n"
        f"Pressing needs: {form_data.get('pressing_needs', [])}\n\n"
        "Return JSON: "
        '{"final_damage_level": "minimal|partial|complete", '
        '"priority_score": 0-100, '
        '"confidence": 0.0-1.0, '
        '"recommended_action": "one sentence action for responders", '
        '"reasoning": "two sentences max"}'
    )

    # OpenAI GPT-4o (primary)
    if OPENAI_API_KEY:
        try:
            r = requests.post(
                OPENAI_URL,
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
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

    # Gemini fallback
    payload = {
        "contents": [{
            "parts": [{"text": f"System: {system_prompt}\n\nUser: {user_content}"}]
        }]
    }
    response = _gemini_post(payload, timeout=20)
    if response:
        result = _extract_json_from_gemini(response)
        if result:
            return result

    # Hard fallback — use reported data + image analysis
    reported = form_data.get("damage_level", "unknown")
    ai_damage = image_analysis.get("damage_level", reported)
    final = ai_damage if ai_damage != "unknown" else reported
    priority = {"minimal": 20, "partial": 55, "complete": 85}.get(final, 30)

    return {
        "final_damage_level": final,
        "priority_score": priority,
        "confidence": image_analysis.get("confidence", 0.3),
        "recommended_action": "Dispatch assessment team to verify damage.",
        "reasoning": "Automated assessment unavailable; using reported values.",
    }


def step4_duplicate_detection(
    building_footprint_id: Optional[str],
    device_id: str,
    report_id: str,
) -> Tuple[bool, Optional[str]]:
    """
    Step 4 — Duplicate detection.
    FIX 3 — guard against None building_footprint_id corrupting all records.
    FIX 8 — version_group_id is assigned when a duplicate IS found, not when it is not.
    Returns (is_duplicate_flagged, version_group_id).
    """
    # FIX 3 — never query or update with an empty building_footprint_id
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

    # FIX 8 — version group only assigned when existing reports found for same building
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


# ── Area signal score ──────────────────────────────────────────────────────────

def compute_area_signal_score(area_signals: dict, combined: dict) -> float:
    score = combined.get("priority_score", 30) / 100
    density_boost = min(area_signals.get("density_per_km2", 0) / 20, 0.3)
    return round(min(score + density_boost, 1.0), 3)


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.route("/api/submit-report", methods=["POST", "OPTIONS"])
def submit_report():
    """
    POST /api/submit-report
    Accepts: form data (all UNDP fields) + optional base64 images
    Returns: submission_id, ai_classification, priority_score
    """
    if request.method == "OPTIONS":
        return options_response()

    logging.info("submit-report called")
    body = request.get_json(silent=True)
    if not body:
        return cors_response(json.dumps({"error": "Invalid JSON"}), 400)

    # FIX 7 — rate limit check before any processing
    device_id = body.get("anonymous_device_id", "")
    if not device_id:
        return cors_response(json.dumps({"error": "anonymous_device_id required"}), 400)
    if is_rate_limited(device_id):
        return cors_response(
            json.dumps({"error": "Rate limit exceeded. Please wait before submitting again."}), 429
        )

    required = [
        "infrastructure_type", "damage_level", "crisis_type",
        "has_debris", "electricity_status", "health_services_status",
        "pressing_needs", "anonymous_device_id",
    ]
    missing = [f for f in required if not body.get(f)]
    if missing:
        return cors_response(
            json.dumps({"error": f"Missing required fields: {missing}"}), 400
        )

    report_id    = str(uuid.uuid4())
    submitted_at = body.get("submitted_at") or datetime.now(timezone.utc).isoformat()
    lat          = body.get("lat")
    lng          = body.get("lng")

    images_b64: List[str] = body.get("images_b64", [])
    image_urls: List[str] = []
    for i, b64 in enumerate(images_b64[:4]):
        url = upload_image_to_storage(b64, report_id, i)
        if url:
            image_urls.append(url)

    if not image_urls:
        image_urls = body.get("image_urls", [])

    image_analysis = step1_image_analysis(images_b64[:4] if images_b64 else [])
    area_signals   = step2_area_signals(
        float(lat) if lat is not None else None,
        float(lng) if lng is not None else None,
    )
    combined       = step3_combined_assessment(image_analysis, area_signals, body)
    is_dup, vg_id  = step4_duplicate_detection(
        body.get("building_footprint_id"), device_id, report_id
    )
    area_score     = compute_area_signal_score(area_signals, combined)

    pressing_needs = body.get("pressing_needs", [])
    if isinstance(pressing_needs, str):
        pressing_needs = [pressing_needs]

    record = {
        "id":                        report_id,
        "submitted_at":              submitted_at,
        "lat":                       float(lat) if lat is not None else None,
        "lng":                       float(lng) if lng is not None else None,
        "building_footprint_id":     body.get("building_footprint_id"),
        "location_description":      body.get("location_description"),
        "infrastructure_type":       body.get("infrastructure_type"),
        "infrastructure_type_other": body.get("infrastructure_type_other"),
        "infrastructure_name":       body.get("infrastructure_name"),
        "damage_level":              body.get("damage_level"),
        "crisis_type":               body.get("crisis_type"),
        "has_debris":                body.get("has_debris"),
        "electricity_status":        body.get("electricity_status"),
        "health_services_status":    body.get("health_services_status"),
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
    }

    sb_insert("reports", record)

    # FIX 8 — only propagate version_group when duplicates confirmed AND footprint exists
    if vg_id and is_dup and body.get("building_footprint_id"):
        sb_update(
            "reports",
            {"building_footprint_id": body.get("building_footprint_id")},
            {"version_group_id": vg_id},
        )

    return cors_response(json.dumps({
        "submission_id":         report_id,
        "ai_classification":     combined.get("final_damage_level"),
        "ai_priority_score":     combined.get("priority_score"),
        "ai_reasoning":          combined.get("reasoning"),
        "ai_recommended_action": combined.get("recommended_action"),
        "area_signals":          area_signals,
        "is_duplicate_flagged":  is_dup,
        "image_urls":            image_urls,
    }), 201)


@app.route("/api/reports", methods=["GET", "OPTIONS"])
def get_reports():
    """
    GET /api/reports
    Returns all reports as GeoJSON with optional filters.
    Query params: crisis_type, damage_level, infrastructure_type,
                  date_from, date_to, bbox (minlng,minlat,maxlng,maxlat),
                  format (geojson|csv)
    """
    if request.method == "OPTIONS":
        return options_response()

    p = request.args
    filters = []

    if p.get("crisis_type"):
        filters.append(f"crisis_type.eq.{p['crisis_type']}")
    if p.get("damage_level"):
        filters.append(f"damage_level.eq.{p['damage_level']}")
    if p.get("infrastructure_type"):
        filters.append(f"infrastructure_type.eq.{p['infrastructure_type']}")
    # FIX 1 — both date_from and date_to applied via and(), neither overwrites the other
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


@app.route("/api/dashboard-stats", methods=["GET", "OPTIONS"])
def dashboard_stats():
    """
    GET /api/dashboard-stats
    Returns aggregate statistics for the dashboard analytics panel.
    """
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
        for r in rows
        if r.get("lat") and r.get("lng")
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


@app.route("/api/export", methods=["POST", "OPTIONS"])
def export_data():
    """
    POST /api/export
    Body: { filters: {...}, format: "csv"|"geojson" }
    Returns filtered data as downloadable CSV or GeoJSON.
    """
    if request.method == "OPTIONS":
        return options_response()

    body       = request.get_json(silent=True) or {}
    filters_in = body.get("filters", {})
    fmt        = body.get("format", "csv")

    filters = []
    if filters_in.get("crisis_type"):
        filters.append(f"crisis_type.eq.{filters_in['crisis_type']}")
    if filters_in.get("damage_level"):
        filters.append(f"damage_level.eq.{filters_in['damage_level']}")
    if filters_in.get("infrastructure_type"):
        filters.append(f"infrastructure_type.eq.{filters_in['infrastructure_type']}")
    # FIX 1 — date range applied correctly via and()
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


@app.route("/api/reporter-alert", methods=["POST", "OPTIONS"])
def reporter_alert():
    """
    POST /api/reporter-alert
    Receives a silent reporter safety alert from the Flutter app.
    No PII — only anonymous_device_id, GPS, and timestamp.
    """
    if request.method == "OPTIONS":
        return options_response()

    body = request.get_json(silent=True)
    if not body:
        return cors_response(json.dumps({"error": "Invalid JSON"}), 400)

    record = {
        "id":                   str(uuid.uuid4()),
        "alert_type":           body.get("alert_type", "REPORTER_SAFETY_ALERT"),
        "anonymous_device_id":  body.get("anonymous_device_id", "unknown"),
        "coordinator_id":       body.get("coordinator_id", ""),
        "lat":                  body.get("lat"),
        "lng":                  body.get("lng"),
        "accuracy_metres":      body.get("accuracy_metres"),
        "triggered_at":         body.get("triggered_at") or datetime.now(timezone.utc).isoformat(),
    }

    sb_insert("reporter_alerts", record)
    logging.info(
        f"Reporter safety alert from {record['anonymous_device_id']} "
        f"to coordinator {record['coordinator_id']}"
    )

    return cors_response(json.dumps({"status": "received"}), 201)


# ── Serialisation helpers ──────────────────────────────────────────────────────

def _rows_to_geojson(rows: list) -> dict:
    features = []
    for r in rows:
        if not r.get("lat") or not r.get("lng"):
            continue
        features.append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(r["lng"]), float(r["lat"])],
            },
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
    "language_submitted", "anonymous_device_id",
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

    # FIX 6 — UTF-8 BOM so Excel correctly renders Arabic, Chinese, and other non-Latin scripts
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
