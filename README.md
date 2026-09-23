# EKT Reorder Agent

This project implements a lightweight procurement recommendation service for warehouse planning. It is designed for the business case of TОО "Электрокомплект" and focuses on realistic inventory logic rather than a heavy frontend.

## Business goal

The purchasing manager needs a fast way to estimate purchase volumes without overstocking or creating shortages. The service calculates what to order by supplier, explains the logic behind the recommendation, and filters out one-off large orders so they do not distort regular demand.

## What is included

- detection of large one-off orders and removal from regular-demand analysis;
- calculation of regular demand from historical sales;
- stockout compensation for lost sales periods;
- supplier and lead-time weighting;
- simple ASCII recommendation dashboard;
- AI-assisted explanation layer using OpenAI when a key is configured.

## Methodology

The logic is intentionally transparent and explainable:

1. historical sales are cleaned from abnormal spikes;
2. regular demand is estimated on the remaining sales history;
3. on-hand stock and inbound quantities are subtracted from the projected need;
4. lead time and safety stock are added;
5. stockout-related lost sales are compensated;
6. the output is grouped by supplier with a short justification.

This follows the core requirement: one-off large orders must not inflate the warehouse replenishment plan.

## Project structure

- `app/` — FastAPI service, config, AI layer and dashboard output
- `src/` — core reorder optimization algorithm
- `data/` — sample CSV inputs for tests and demo runs
- `tests/` — validation of logic, API auth and dashboard output

## Run locally

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Then open:

- http://localhost:8000/
- http://localhost:8000/health
- http://localhost:8000/docs
- http://localhost:8000/api/v1/recommendations/ascii

## Authentication

All recommendation endpoints use bearer-token auth.

Default token:

```bash
change-me-token
```

Override it via `APP_API_TOKEN` in `.env`.

## OpenAI configuration

Create a `.env` file in the project root with:

```env
APP_API_TOKEN=change-me-token
OPENAI_API_KEY=your_openai_key_here
```

The app will read the key automatically when available. If the key is missing, the service falls back to a non-AI summary format so the project remains usable.

## Simple ASCII dashboard

The service exposes a textual dashboard that is intentionally minimal and easy to view in a terminal or browser:

```text
article      | supplier         | recommended_qty  | reason
------------+------------------+------------------+----------------------------------------
EL-100       | AlphaParts       | 270.6            | avg_demand=12.0; lead_time=10d...
```

This keeps the interface very lightweight while still supporting operational review.

## Validation

```bash
pytest -q
```

Current validation status:

- 4 passed

## Notes

- The project is designed for local/demo operation first and can later be deployed to a server or hosting provider.
- The logic is explainable and suitable for managerial review before approving purchase orders.
- The design intentionally avoids a heavy frontend, as requested for a simple operational tool.

