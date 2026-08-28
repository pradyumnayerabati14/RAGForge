from fastapi.testclient import TestClient

from ragforge.api import app
from ragforge.container import get_container


def test_health_and_qa_flow():
    get_container.cache_clear()
    with TestClient(app) as client:
        assert client.get("/health").json()["status"] == "ok"
        response = client.post("/v1/documents", json={"documents": [{
            "id": "python", "text": "Python was created by Guido van Rossum.", "source": "facts.md"
        }]})
        assert response.status_code == 200
        response = client.post("/v1/ask", json={"question": "Who created Python?"})
        assert response.status_code == 200
        assert response.json()["citations"][0]["document_id"] == "python"

