from datetime import datetime
import pytest


def overit_studenta(student_data):
    vek_limit = 18
    prumer_limit = 2.5
    deadline = datetime(2025, 5, 15)

    if student_data["vek"] < vek_limit:
        return "Student neni plnolety"
    if student_data["prumer"] < prumer_limit: 
        return "Student nema dostatecny prumer"
    if student_data["datum_podani"] >= deadline: 
        return "Student nepodal vcas"

    return "Student je platny"


@pytest.mark.parametrize("student_data, expected_result", [
    
    ({"vek": 18, "prumer": 2.5, "datum_podani": datetime(2025, 5, 14)}, "Student je platny"), 
    ({"vek": 17, "prumer": 2.5, "datum_podani": datetime(2025, 5, 14)}, "Student neni plnolety"), 
    ({"vek": 18, "prumer": 2.4, "datum_podani": datetime(2025, 5, 14)}, "Student nema dostatecny prumer"), 
    ({"vek": 18, "prumer": 2.5, "datum_podani": datetime(2025, 5, 15)}, "Student nepodal vcas"), 
    ({"vek": 18, "prumer": 2.5, "datum_podani": datetime(2025, 5, 16)}, "Student nepodal vcas"), 
    ({"vek": 18, "prumer": 3.0, "datum_podani": datetime(2025, 5, 14)}, "Student je platny"), 
])

def test_overit_studenta(student_data, expected_result):
    assert overit_studenta(student_data) == expected_result