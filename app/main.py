from __future__ import annotations

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.responses import HTMLResponse

from app.agent import ReorderAgent
from app.ai_service import build_ai_summary
from app.config import API_TOKEN
from app.visuals import render_ascii_dashboard

app = FastAPI(title="EKT Reorder Agent", version="1.0.0")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <html>
      <head>
        <title>EKT Reorder Agent</title>
        <style>
          body { font-family: Arial, sans-serif; margin: 40px; background: #f7f9fc; color: #1f2937; }
          .card { max-width: 700px; margin: auto; background: white; border-radius: 12px; padding: 24px; box-shadow: 0 10px 30px rgba(0,0,0,0.08); }
          code { background: #eef2ff; padding: 2px 6px; border-radius: 4px; }
        </style>
      </head>
      <body>
        <div class="card">
          <h1>EKT Reorder Agent</h1>
          <p>Сервис рекомендаций закупок запущен.</p>
          <p><a href="/docs">Open API docs</a></p>
          <p>Health: <code>/health</code></p>
          <p>Recommendation endpoint: <code>/api/v1/recommendations/calculate</code></p>
        </div>
      </body>
    </html>
    """


def require_token(authorization: str | None = Header(default=None, alias="Authorization")) -> str:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid bearer token")

    token = authorization.split(" ", 1)[1].strip()
    if token != API_TOKEN:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    return token


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "ekt-reorder-agent"}


@app.post("/api/v1/agent/train")
def train_agent(token: str = Depends(require_token)) -> dict:
    agent = ReorderAgent()
    training_summary = agent.fit()
    validation_summary = agent.validate()
    return {
        "token_valid": bool(token),
        "training_summary": training_summary,
        "validation_summary": validation_summary,
    }


@app.post("/api/v1/agent/validate")
def validate_agent(token: str = Depends(require_token)) -> dict:
    agent = ReorderAgent()
    validation_summary = agent.validate()
    return {"token_valid": bool(token), "validation_summary": validation_summary}


@app.post("/api/v1/recommendations/calculate")
def calculate_recommendations(token: str = Depends(require_token)) -> dict:
    agent = ReorderAgent()
    recommendations = agent.calculate()
    validation_summary = agent.validate()
    return {
        "token_valid": bool(token),
        "recommendations": recommendations,
        "validation_summary": validation_summary,
        "ai_summary": build_ai_summary(recommendations),
        "ascii_dashboard": render_ascii_dashboard(recommendations),
    }


@app.get("/api/v1/recommendations/ascii")
def ascii_dashboard(token: str = Depends(require_token)) -> str:
    agent = ReorderAgent()
    recommendations = agent.calculate()
    return render_ascii_dashboard(recommendations)
