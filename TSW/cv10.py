import requests
import pytest

BASE_URL = "http://127.0.0.1:5000/validate"


test_data = [
    ({"age": 1, "citizenship": "EU"}, 200, {"valid": True, "message": "Passenger data is valid."}),
    ({"age": 30, "citizenship": "EU", "document_type": "ID card"}, 200, {"valid": True, "message": "Passenger data is valid."}),
    ({"age": 40, "citizenship": "EU", "document_type": "passport"}, 200, {"valid": True, "message": "Passenger data is valid."}),
    ({"age": 50, "citizenship": "non-EU", "document_type": "passport"}, 200, {"valid": True, "message": "Passenger data is valid."}),    
    ({"age": 25, "citizenship": "EU"}, 400, {"valid": False, "errors": ["Missing 'document_type' parameter for age >= 2."]}),
    ({"age": -1, "citizenship": "EU", "document_type": "ID card"}, 400, {"valid": False, "errors": ["Invalid age. Must be between 0 and 120."]}),
    ({"age": 121, "citizenship": "EU", "document_type": "ID card"}, 400, {"valid": False, "errors": ["Invalid age. Must be between 0 and 120."]}),
    ({"age": 30, "citizenship": "non-EU", "document_type": "ID card"}, 400, {"valid": False, "errors": ["Non-EU citizens must have a passport."]}),

]

@pytest.mark.parametrize("params, expected_status, expected_json", test_data)
def test_validate_passenger(params, expected_status, expected_json):
    """
    Tests the /validate endpoint with various parameter combinations
    based on equivalence partitioning.
    """
    
    query_params = {k: v for k, v in params.items() if v is not None}
    response = requests.get(BASE_URL, params=query_params)

    assert response.status_code == expected_status
    
    
    response_data = response.json()
    if "errors" in expected_json:
         assert "errors" in response_data
         
         assert all(item in response_data["errors"] for item in expected_json["errors"])
         
         assert response_data.get("valid") is False
    else:
         assert response_data == expected_json


