from datetime import datetime
import pytest


def overit_studenta(student_data):
    vek_limit = 18
    prumer_limit = 2.5
    deadline = datetime(2025, 5, 15)

    if student_data["vek"] < vek_limit:
        return "Student je neplatny"
    
    if student_data["prumer"] > prumer_limit: 
        return "Student je neplatny"
    
    if student_data["datum_podani"] > deadline: 
        return "Student je neplatny"

    return "Student je platny"


@pytest.mark.parametrize("student_data, expected_result", [
    ({"vek": 17, "prumer": 2.5, "datum_podani": datetime(2025, 5, 14)}, "Student je neplatny"), 
    ({"vek": 18, "prumer": 2.5, "datum_podani": datetime(2025, 5, 14)}, "Student je platny"), 
    ({"vek": 19, "prumer": 2.5, "datum_podani": datetime(2025, 5, 14)}, "Student je platny"), 


    ({"vek": 19, "prumer": 2.6, "datum_podani": datetime(2025, 5, 14)}, "Student je neplatny"), 
    ({"vek": 19, "prumer": 2.5, "datum_podani": datetime(2025, 5, 14)}, "Student je platny"), 
    ({"vek": 19, "prumer": 2.4, "datum_podani": datetime(2025, 5, 14)}, "Student je platny"), 

    ({"vek": 19, "prumer": 1.0, "datum_podani": datetime(2025, 5, 16)}, "Student je neplatny"), 
    ({"vek": 19, "prumer": 1.0, "datum_podani": datetime(2025, 5, 15)}, "Student je platny"), 
    ({"vek": 19, "prumer": 1.0, "datum_podani": datetime(2025, 5, 14)}, "Student je platny"), 


])

def test_overit_studenta(student_data, expected_result):
    assert overit_studenta(student_data) == expected_result, f"Test failed for student data: {student_data}"