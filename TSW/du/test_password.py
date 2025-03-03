from password import is_valid_password
import pytest

@pytest.fixture
def valid_password_checker():
    return is_valid_password

@pytest.mark.parametrize("password, result", [
    pytest.param("Aa123456789", True, id="valid password"),
    pytest.param("aaa", False, id="too short"),
    pytest.param("aaaaaaaaaaaaaaaaaaa", False, id="missing num"),
    pytest.param("aaaaaaa6aaaaaaaaaaaa", False, id="missing upper"),
    pytest.param("AAAAAAAAAAAAAAAAAA666", False, id="missing lower"),
    pytest.param("66666666666666666666666666", False, id="missing alpha"),
])
def test_add(valid_password_checker, password, result):
    assert valid_password_checker(password) == result