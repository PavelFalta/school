from cv3 import Kalkulacka

def test_multiply(calc):
    a, b, c, d, expected = 1, 2, 6, 3, 36

    outcome = calc.multiply(a, b, c, d)

    assert outcome == expected


calc = Kalkulacka()

test_multiply(calc)