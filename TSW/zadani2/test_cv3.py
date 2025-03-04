from cv3 import Kalkulacka

def test_multiply(calc):
    a, b, c, d, expected = float('inf'), 2, 6, 3, float('inf')

    outcome = calc.multiply(a, b, c, d)

    assert outcome == expected


calc = Kalkulacka()

test_multiply(calc)