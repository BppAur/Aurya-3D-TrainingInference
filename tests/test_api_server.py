import pytest
from fastapi.testclient import TestClient
from scripts.api_server import app

@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint(client):
    """Test that health endpoint returns 200"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}
