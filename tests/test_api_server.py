import pytest
from fastapi.testclient import TestClient
from scripts.api_server import app, TEMP_DIR, MAX_FILE_SIZE, validate_path_in_temp_dir
import tempfile
from pathlib import Path
import asyncio
import time

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

def test_file_size_limit_image(client):
    """Test that large image files are rejected"""
    # Create a file larger than 100MB
    large_data = b"x" * (101 * 1024 * 1024)  # 101MB
    small_mesh = b"small mesh data"

    response = client.post(
        "/infer",
        files={
            "image": ("large.png", large_data, "image/png"),
            "coarse_mesh": ("test.glb", small_mesh, "model/gltf-binary")
        }
    )

    assert response.status_code == 413  # Request Entity Too Large
    assert "too large" in response.json()["detail"].lower()

def test_file_size_limit_mesh(client):
    """Test that large mesh files are rejected"""
    small_image = b"small image data"
    large_mesh = b"x" * (101 * 1024 * 1024)  # 101MB

    response = client.post(
        "/infer",
        files={
            "image": ("test.png", small_image, "image/png"),
            "coarse_mesh": ("large.glb", large_mesh, "model/gltf-binary")
        }
    )

    assert response.status_code == 413  # Request Entity Too Large
    assert "too large" in response.json()["detail"].lower()

def test_path_traversal_validation():
    """Test that path traversal is detected and prevented"""
    # Test valid path within TEMP_DIR
    valid_path = TEMP_DIR / "test.glb"
    validated = validate_path_in_temp_dir(valid_path)
    assert str(validated).startswith(str(TEMP_DIR.resolve()))

    # Test path traversal attempt
    with pytest.raises(ValueError, match="Path traversal detected"):
        traversal_path = TEMP_DIR / ".." / ".." / "etc" / "passwd"
        validate_path_in_temp_dir(traversal_path)

def test_invalid_image_extension(client):
    """Test that invalid image file extensions are rejected"""
    response = client.post(
        "/infer",
        files={
            "image": ("test.txt", b"data", "text/plain"),
            "coarse_mesh": ("test.glb", b"data", "model/gltf-binary")
        }
    )

    assert response.status_code == 400
    assert "Image must be PNG or JPEG" in response.json()["detail"]

def test_invalid_mesh_extension(client):
    """Test that invalid mesh file extensions are rejected"""
    response = client.post(
        "/infer",
        files={
            "image": ("test.png", b"data", "image/png"),
            "coarse_mesh": ("test.obj", b"data", "application/octet-stream")
        }
    )

    assert response.status_code == 400
    assert "Mesh must be GLB format" in response.json()["detail"]

def test_concurrent_requests_semaphore_exists(client):
    """Test that the API has concurrency control via semaphore"""
    # Just verify the semaphore exists and is configured correctly
    from scripts.api_server import inference_semaphore
    assert inference_semaphore is not None
    # Semaphore should allow only 1 concurrent request
    assert inference_semaphore._value == 1, "Semaphore should be configured for 1 concurrent request"

def test_error_message_sanitization(client, monkeypatch):
    """Test that error messages don't expose internal paths or stderr"""
    import subprocess

    def mock_run(*args, **kwargs):
        # Simulate a subprocess error with sensitive info in stderr
        raise subprocess.CalledProcessError(
            1,
            args[0],
            stderr="/internal/secret/path/error: file not found"
        )

    original_exists = Path.exists
    def mock_exists(self):
        path_str = str(self)
        if "checkpoint" in path_str or "config" in path_str:
            return True
        return original_exists(self)

    monkeypatch.setattr("subprocess.run", mock_run)
    monkeypatch.setattr(Path, "exists", mock_exists)

    response = client.post(
        "/infer",
        files={
            "image": ("test.png", b"fake image", "image/png"),
            "coarse_mesh": ("test.glb", b"fake mesh", "model/gltf-binary")
        }
    )

    assert response.status_code == 500
    # Error message should be sanitized
    detail = response.json()["detail"]
    assert "/internal/secret/path" not in detail
    assert "please check input files" in detail.lower()

def test_missing_checkpoint(client, monkeypatch):
    """Test that missing checkpoint returns 503"""
    # Don't mock Path.exists - let it check real files
    response = client.post(
        "/infer",
        files={
            "image": ("test.png", b"fake image", "image/png"),
            "coarse_mesh": ("test.glb", b"fake mesh", "model/gltf-binary")
        }
    )

    assert response.status_code == 503
    assert "checkpoint" in response.json()["detail"].lower() or "configuration" in response.json()["detail"].lower()
