from password import is_valid_password
import pytest

@pytest.fixture
def is_valid_password():
    return is_valid_password()


@pytest.mark.parametrize("password, result", [
    pytest.param("Aa123456789", True, id="valid password"),

])
def test_add(is_valid_password, password, result):
    assert is_valid_password(password) == result