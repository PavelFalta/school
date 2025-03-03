from password import is_valid_password
import pytest

@pytest.fixture
def valid_password_checker():
    return is_valid_password

@pytest.mark.parametrize("password, result", [
    pytest.param("Aa123456789", True, id="valid password"),
])
def test_add(valid_password_checker, password, result):
    assert valid_password_checker(password) == result