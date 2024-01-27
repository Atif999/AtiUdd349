import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

test_item = {"text": "Lebensmittel kommssionierung"}

def test_predict_endpoint():

    response = client.post("/predict", json=test_item)
    
    assert response.status_code == 200
    
    assert "predicted_label" in response.json()
    assert "input_text" in response.json()
    
    assert isinstance(response.json()["predicted_label"], list)
    
    assert response.json()["input_text"] == test_item["text"]
