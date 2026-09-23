from fastapi.testclient import TestClient

from app.main import app


def test_openapi_uses_bearer_security_scheme() -> None:
    client = TestClient(app)
    schema = client.get("/openapi.json").json()

    assert "components" in schema
    assert "securitySchemes" in schema["components"]
    assert "HTTPBearer" in schema["components"]["securitySchemes"]
    security_scheme = schema["components"]["securitySchemes"]["HTTPBearer"]
    assert security_scheme["type"] == "http"
    assert security_scheme["scheme"] == "bearer"

    recommendations = schema["paths"]["/api/v1/recommendations/calculate"]["post"]
    assert recommendations.get("security") == [{"HTTPBearer": []}]


def test_root_page_has_calculator_ui() -> None:
    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    html = response.text
    assert "EKT Reorder Agent" in html
    assert "Calculate Recommendations" in html
    assert "Authorization" in html
    assert "fetch(" in html
    assert "Шаг 1" in html
    assert "Введите токен" in html
