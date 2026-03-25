# CrisisMap — Backend + Dashboard

UNDP Crisis Damage Reporting — Azure Functions backend and responder web dashboard.
Converted from SilentPulse. MIT Licence.

---

## Architecture

```
Flutter App (offline-first)
    ↓ base64 images + UNDP form data
POST /api/submit-report
    ├── Upload images → Supabase Storage
    ├── Step 1: Gemini Vision → image damage analysis
    ├── Step 2: Supabase query → area signals (500m / 2h)
    ├── Step 3: Gemini/GPT-4o → combined priority assessment
    ├── Step 4: Duplicate detection → version_group_id
    └── INSERT → Supabase reports table
         ↓
GET /api/reports         → GeoJSON for map
GET /api/dashboard-stats → Analytics panel data
POST /api/export         → CSV or GeoJSON download
POST /api/reporter-alert → Silent reporter safety alert storage

Web Dashboard (index.html)
    ├── Leaflet.js + OpenStreetMap (no Azure Maps key needed)
    ├── Marker clustering + heatmap toggle
    ├── Click marker → AI insight panel (photos, reasoning, version history)
    ├── Analytics: Chart.js doughnut/bar/line charts
    ├── Priority queue: top 10 by AI priority score
    ├── Export: CSV + GeoJSON respecting active filters
    └── Language toggle: EN / AR / ZH / FR / RU / ES
```

---

## Required environment variables

Set these in Azure Function App → Configuration → Application Settings:

| Variable | Required | Description |
|---|---|---|
| `SUPABASE_URL` | Yes | e.g. `https://xxxx.supabase.co` |
| `SUPABASE_ANON_KEY` | Yes | Supabase anon or service_role key |
| `GEMINI_API_KEY` | Yes | Google Gemini API key (image analysis + fallback) |
| `OPENAI_API_KEY` | Yes | OpenAI API key (Step 3 combined assessment, GPT-4o) |

**Never hardcode these values.**

---

## Database setup (Supabase)

Run the SQL schema block at the bottom of `function_app.py` in the Supabase SQL Editor. It creates:

- `reports` table (all UNDP fields + AI analysis columns)
- `reporter_alerts` table (anonymous GPS safety alerts)
- Storage bucket `report-images` (public read, anonymous write)
- Row Level Security policies
- Indexes on lat/lng, submitted_at, damage_level, building_footprint_id

---

## Run locally

```bash
cd Silentpulse
pip install -r requirements.txt
func start
```

Set env vars in `local.settings.json`:
```json
{
  "IsEncrypted": false,
  "Values": {
    "AzureWebJobsStorage": "UseDevelopmentStorage=true",
    "FUNCTIONS_WORKER_RUNTIME": "python",
    "SUPABASE_URL": "https://xxxx.supabase.co",
    "SUPABASE_ANON_KEY": "...",
    "GEMINI_API_KEY": "...",
    "OPENAI_API_KEY": "..."
  }
}
```

---

## Dashboard

Open `dashboard/index.html` in a browser, or deploy to any static host.

The dashboard reads from:
```javascript
const API_BASE = window.API_BASE_URL || 'https://your-azure-function-app.azurewebsites.net/api';
```

To configure without editing the HTML, serve with a proxy that sets `window.API_BASE_URL`, or add a `<script>` tag to `index.html`:
```html
<script>window.API_BASE_URL = 'https://your-app.azurewebsites.net/api';</script>
```

---

## API endpoints

| Method | Route | Description |
|---|---|---|
| POST | `/api/submit-report` | Receive report + images; run AI pipeline; store to Supabase |
| GET | `/api/reports` | All reports as GeoJSON; supports filters + bbox + format=csv |
| GET | `/api/dashboard-stats` | Aggregate stats, heatmap points, priority queue |
| POST | `/api/export` | Filtered CSV or GeoJSON download |
| POST | `/api/reporter-alert` | Store anonymous reporter safety alert |

All endpoints include CORS headers for cross-origin dashboard access.

---

## AI pipeline (per submission)

1. **Gemini Vision** — analyses all images, classifies damage level, detects debris, returns confidence
2. **Area signals** — queries Supabase for reports within 500m in last 2 hours; calculates average damage and dominant crisis type
3. **Combined assessment** — sends image result + area signals + form data to OpenAI GPT-4o (primary), Gemini as fallback; returns priority score 0–100 + recommended action
4. **Duplicate detection** — checks building_footprint_id; links version group; flags if same device submitted within 10 minutes

---

## No PII

- `anonymous_device_id` is a UUID4 generated on the device — never linked to a person
- No names, phone numbers, or email addresses are stored or processed
- Reporter safety alerts store only: device ID, coordinator ID, GPS, timestamp
