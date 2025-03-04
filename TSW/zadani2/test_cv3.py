from cv3 import Kalkulacka

def test_multiply(calc):
    a, expected = [1,2,3], 6
    outcome = calc.multiply(a)
    assert outcome == expected


calc = Kalkulacka()


test_multiply(calc)