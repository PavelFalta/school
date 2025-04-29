from datetime import datetime
import pytest

@pytest.mark.parametrize("student_data, expected_result", [
    ({"vek": 18, "prumer": 2.5, "datum_podani": datetime(2025, 5, 15)}, "Student je platny"),
    ({"vek": 17, "prumer": 2.5, "datum_podani": datetime(2025, 5, 15)}, "Student neni plnolety"),
    ({"vek": 18, "prumer": 2.4, "datum_podani": datetime(2025, 5, 15)}, "Student nema dostatecny prumer"),
    ({"vek": 18, "prumer": 2.5, "datum_podani": datetime(2025, 5, 16)}, "Student nepodal vcas"),
])
def zapsat_na_vejsku(student_data, expected_result):
    assert zapsat_na_vejsku(student_data) == expected_result

    if student_data["prumer"] > 2.5:
        return "Student nema dostatecny prumer"
    if student_data["datum_podani"] >= datetime(2025, 5, 15):
        return "Student nepodal vcas"
    
    return "Student je platny"

student_data = {
    "vek": 20,
    "prumer": 1.5,
    "datum_podani": datetime(2025, 1, 1),
}

print(zapsat_na_vejsku(student_data))