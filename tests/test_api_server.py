import pytest
from fastapi.testclient import TestClient
from scripts.api_server import app
import tempfile
from pathlib import Path

@pytest.fixture
def client():
    return TestClient(app)

def test_health_endpoint(client):
    """Test that health endpoint returns 200"""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_inference_endpoint_missing_files(client):
    """Test that inference endpoint validates required files"""
    response = client.post("/infer")
    assert response.status_code == 422  # Missing required fields

def test_inference_endpoint_with_files(client, monkeypatch):
    """Test that inference endpoint accepts files and returns mesh"""
    # Mock the inference function to avoid requiring actual model
    def mock_infer(*args, **kwargs):
        output_path = args[2] if len(args) > 2 else kwargs.get("output_path", "/tmp/output.glb")
        Path(output_path).write_text("mock mesh data")
        return str(output_path)

    # Mock Path.exists to return True for checkpoint and config files
    original_exists = Path.exists
    def mock_exists(self):
        path_str = str(self)
        if "checkpoint" in path_str or "config" in path_str:
            return True
        return original_exists(self)

    monkeypatch.setattr("scripts.api_server.run_inference", mock_infer)
    monkeypatch.setattr(Path, "exists", mock_exists)

    with tempfile.NamedTemporaryFile(suffix=".png") as img_file:
        with tempfile.NamedTemporaryFile(suffix=".glb") as mesh_file:
            img_file.write(b"fake image data")
            mesh_file.write(b"fake mesh data")
            img_file.seek(0)
            mesh_file.seek(0)

            response = client.post(
                "/infer",
                files={
                    "image": ("test.png", img_file, "image/png"),
                    "coarse_mesh": ("test.glb", mesh_file, "model/gltf-binary")
                }
            )

            assert response.status_code == 200
            assert response.headers["content-type"] == "model/gltf-binary"
