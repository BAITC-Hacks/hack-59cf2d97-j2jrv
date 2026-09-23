from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("BLIS_NUM_THREADS", "1")

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.responses import HTMLResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.agent import ReorderAgent
from app.ai_service import build_ai_summary
from app.config import API_TOKEN
from app.visuals import render_ascii_dashboard

security = HTTPBearer(auto_error=False)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    get_cached_recommendations()
    yield


app = FastAPI(title="EKT Reorder Agent", version="1.0.0", lifespan=lifespan)
MAX_RECOMMENDATIONS = 25


@lru_cache(maxsize=1)
def get_cached_recommendations() -> tuple[list[dict], dict]:
    agent = ReorderAgent()
    all_recommendations = agent.calculate()
    recommendations = sorted(
        all_recommendations,
        key=lambda item: float(item.get("recommended_qty", 0) or 0),
        reverse=True,
    )[:MAX_RECOMMENDATIONS]
    validation_summary = agent.validate()
    return recommendations, validation_summary


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return """
    <!doctype html>
    <html lang="ru">
      <head>
        <meta charset="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <title>EKT Reorder Agent</title>
        <style>
          :root {
            --bg: #f3f7fb;
            --panel: #ffffff;
            --panel-alt: #edf5f2;
            --primary: #1ca98b;
            --primary-dark: #107d66;
            --text: #16212c;
            --muted: #586876;
            --error: #c94747;
            --shadow: 0 18px 38px rgba(15, 38, 54, 0.12);
            --radius: 18px;
          }

          * { box-sizing: border-box; }

          body {
            margin: 0;
            font-family: "Segoe UI", Arial, sans-serif;
            background: linear-gradient(180deg, #ebf4f6 0%, var(--bg) 100%);
            color: var(--text);
          }

          .page {
            max-width: 1200px;
            margin: 0 auto;
            padding: 32px 18px 60px;
          }

          .header-card {
            background: linear-gradient(135deg, #1aa77f 0%, #5bc6ba 100%);
            color: white;
            border-radius: var(--radius);
            padding: 24px 28px;
            box-shadow: var(--shadow);
            margin-bottom: 20px;
          }

          .header-card h1 {
            margin: 0 0 8px;
            font-size: 2.4rem;
          }

          .header-card p {
            margin: 0;
            opacity: 0.96;
            font-size: 1.03rem;
          }

          .workspace {
            display: grid;
            gap: 22px;
            grid-template-columns: minmax(300px, 380px) minmax(0, 1fr);
          }

          .panel {
            background: var(--panel);
            border-radius: var(--radius);
            box-shadow: var(--shadow);
            padding: 22px;
          }

          .form-title, .section-title {
            margin: 0 0 18px;
            font-size: 1.2rem;
            font-weight: 700;
          }

          label {
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
          }

          input {
            width: 100%;
            border: 1px solid #d5dfe8;
            border-radius: 10px;
            padding: 12px 14px;
            font-size: 1rem;
            background: #fbfdff;
            color: var(--text);
          }

          input:focus {
            outline: none;
            border-color: var(--primary);
            box-shadow: 0 0 0 3px rgba(28, 169, 139, 0.12);
          }

          .btn {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            border: none;
            border-radius: 10px;
            background: var(--primary);
            color: white;
            padding: 12px 18px;
            font-weight: 700;
            font-size: 0.98rem;
            cursor: pointer;
            transition: transform 0.15s ease, opacity 0.15s ease;
          }

          .btn:hover {
            transform: translateY(-1px);
            opacity: 0.98;
          }

          .btn:disabled {
            opacity: 0.7;
            cursor: wait;
          }

          .secondary {
            background: #e8f4ef;
            color: var(--primary-dark);
          }

          .stack {
            display: grid;
            gap: 14px;
          }

          .helper {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
          }

          .error {
            display: none;
            background: #ffe7e7;
            color: var(--error);
            border: 1px solid #f5c4c4;
            border-radius: 10px;
            padding: 12px 14px;
            font-weight: 600;
          }

          .error.visible {
            display: block;
          }

          .status {
            color: var(--muted);
            min-height: 24px;
            font-size: 0.94rem;
            font-weight: 600;
          }

          .metrics {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 14px;
            margin-top: 12px;
          }

          .metric {
            background: var(--panel-alt);
            padding: 16px 14px;
            border-radius: 12px;
            border: 1px solid #dfece8;
          }

          .metric small {
            display: block;
            color: var(--muted);
            margin-bottom: 8px;
          }

          .metric strong {
            font-size: 1.2rem;
          }

          table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 6px;
            background: white;
          }

          th, td {
            border-bottom: 1px solid #edf0f3;
            padding: 10px 12px;
            text-align: left;
            vertical-align: top;
            font-size: 0.94rem;
          }

          th {
            background: #f7faf9;
            color: var(--text);
            font-weight: 700;
          }

          tbody tr:hover {
            background: #f8fbfc;
          }

          .summary-block {
            background: #f7faf9;
            border: 1px solid #e4ece7;
            border-radius: 12px;
            padding: 16px;
            margin-top: 16px;
            color: var(--text);
          }

          .summary-block h3 {
            margin: 0 0 12px;
            font-size: 1.05rem;
          }

          .guide {
            background: linear-gradient(180deg, #f6fbfb 0%, #eef6f4 100%);
            border: 1px solid #dfeeea;
            border-radius: 12px;
            padding: 14px 16px;
            margin-bottom: 18px;
          }

          .guide ol {
            margin: 12px 0 0 18px;
            padding: 0;
            color: var(--text);
            line-height: 1.7;
          }

          pre {
            white-space: pre-wrap;
            word-break: break-word;
            font-family: "Consolas", "Monaco", monospace;
            background: #121d2b;
            color: #eaf4ff;
            border-radius: 12px;
            padding: 14px;
            margin-top: 16px;
            overflow: auto;
          }

          @media (max-width: 820px) {
            .workspace {
              grid-template-columns: 1fr;
            }

            .metrics {
              grid-template-columns: 1fr;
            }
          }
        </style>
      </head>
      <body>
        <div class="page">
          <div class="header-card">
            <h1>EKT Reorder Agent</h1>
            <p>Сервис рекомендаций по закупкам и мониторинг запасов</p>
          </div>

          <div class="workspace">
            <div class="panel">
              <h2 class="form-title">Calculate Recommendations</h2>

              <div class="guide">
                <strong>Как пользоваться:</strong>
                <ol>
                  <li><strong>Шаг 1:</strong> Введите токен авторизации в поле ниже.</li>
                  <li><strong>Шаг 2:</strong> Нажмите кнопку «Сформировать рекомендации».</li>
                  <li><strong>Шаг 3:</strong> Проверьте приоритет по SKU, остаток на складе и итоговый AI-summary.</li>
                </ol>
              </div>

              <form id="calc-form" class="stack">
                <div>
                  <label for="token">Authorization token</label>
                  <input id="token" type="password" value="change-me-token" placeholder="Введите токен" />
                </div>

                <button class="btn" id="submit-btn" type="submit">Сформировать рекомендации</button>
              </form>

              <div class="helper" style="margin-top: 12px;">
                Используйте токен <strong>change-me-token</strong> по умолчанию или задайте свой в переменной <strong>APP_API_TOKEN</strong>.
              </div>

              <div id="error" class="error" aria-live="polite"></div>
              <div id="status" class="status" aria-live="polite"></div>
            </div>

            <div class="panel">
              <h2 class="section-title">Результаты</h2>

              <div id="metrics" class="metrics"></div>

              <div class="summary-block">
                <h3>AI Summary</h3>
                <div id="ai-summary">Появится после расчёта: приоритет закупки, риск дефицита и краткая рекомендация по действиям.</div>
              </div>

              <div style="margin-top: 18px; overflow: auto;">
                <table>
                  <thead>
                    <tr>
                      <th>Article</th>
                      <th>Supplier</th>
                      <th>Recommended qty</th>
                      <th>Estimated demand</th>
                      <th>Stock on hand</th>
                      <th>Inbound qty</th>
                    </tr>
                  </thead>
                  <tbody id="recommendations-body">
                    <tr>
                      <td colspan="6" style="color: var(--muted);">Пусто. Нажмите кнопку, чтобы сформировать рекомендации.</td>
                    </tr>
                  </tbody>
                </table>
              </div>

              <pre id="raw-json">{}</pre>
            </div>
          </div>
        </div>

        <script>
          const form = document.getElementById('calc-form');
          const tokenInput = document.getElementById('token');
          const submitBtn = document.getElementById('submit-btn');
          const errorBox = document.getElementById('error');
          const statusBox = document.getElementById('status');
          const metricsBox = document.getElementById('metrics');
          const aiSummaryBox = document.getElementById('ai-summary');
          const tableBody = document.getElementById('recommendations-body');
          const rawJsonBox = document.getElementById('raw-json');

          function showError(message) {
            errorBox.textContent = message;
            errorBox.classList.add('visible');
          }

          function clearError() {
            errorBox.textContent = '';
            errorBox.classList.remove('visible');
          }

          function renderMetrics(validation) {
            if (!validation) {
              metricsBox.innerHTML = '';
              return;
            }

            const items = [
              ['Train rows', validation.train_rows ?? 0],
              ['MAE', Number(validation.mae ?? 0).toFixed(2)],
              ['MAPE', `${Number(validation.mape ?? 0).toFixed(2)}%`],
            ];

            metricsBox.innerHTML = items.map(([label, value]) => `
              <div class="metric">
                <small>${label}</small>
                <strong>${value}</strong>
              </div>
            `).join('');
          }

          function renderRows(recommendations) {
            if (!recommendations || !recommendations.length) {
              tableBody.innerHTML = '<tr><td colspan="6" style="color: var(--muted);">Данных нет.</td></tr>';
              return;
            }

            tableBody.innerHTML = recommendations.map(item => `
              <tr>
                <td>${item.article ?? '-'}</td>
                <td>${item.supplier ?? '-'}</td>
                <td>${Number(item.recommended_qty ?? 0).toFixed(2)}</td>
                <td>${Number(item.estimated_demand ?? 0).toFixed(2)}</td>
                <td>${Number(item.stock_on_hand ?? 0)}</td>
                <td>${Number(item.inbound_qty ?? 0)}</td>
              </tr>
            `).join('');
          }

          form.addEventListener('submit', async (event) => {
            event.preventDefault();
            clearError();
            const token = tokenInput.value.trim();

            if (!token) {
              showError('Введите токен авторизации.');
              return;
            }

            submitBtn.disabled = true;
            statusBox.textContent = 'Расчёт запущен...';

            try {
              const response = await fetch('/api/v1/recommendations/calculate', {
                method: 'POST',
                headers: {
                  'Authorization': `Bearer ${token}`,
                  'Accept': 'application/json'
                }
              });

              const payload = await response.json();

              if (!response.ok) {
                throw new Error(payload.detail || 'Ошибка запроса');
              }

              renderMetrics(payload.validation_summary);
              renderRows(payload.recommendations);
              aiSummaryBox.textContent = payload.ai_summary || 'Сводка отсутствует.';
              rawJsonBox.textContent = JSON.stringify(payload, null, 2);
              statusBox.textContent = 'Готово. Данные обновлены.';
            } catch (error) {
              showError(error.message || 'Не удалось выполнить запрос.');
              statusBox.textContent = 'Ошибка';
            } finally {
              submitBtn.disabled = false;
            }
          });
        </script>
      </body>
    </html>
    """


def require_token(credentials: HTTPAuthorizationCredentials | None = Depends(security)) -> str:
    if credentials is None or not credentials.credentials:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid bearer token")

    token = credentials.credentials
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
    recommendations, validation_summary = get_cached_recommendations()
    return {
        "token_valid": bool(token),
        "recommendations": recommendations,
        "validation_summary": validation_summary,
        "ai_summary": build_ai_summary(recommendations),
        "ascii_dashboard": render_ascii_dashboard(recommendations),
    }


@app.get("/api/v1/recommendations/ascii")
def ascii_dashboard(token: str = Depends(require_token)) -> str:
    recommendations, _ = get_cached_recommendations()
    return render_ascii_dashboard(recommendations)
