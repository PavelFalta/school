from TSW.zadani1.calculator import Calculator
import unittest


# AAA - Arrange, Act, Assert

class TestCalculator(unittest.TestCase):
    def setUp(self):
        self.calc = Calculator()
    
    def test_add_two_positive_integers(self):
        # arrange
        a = 2
        b = 3
        expected = 5

        # act
        result = self.calc.add(a, b)

        # assert
        self.assertEqual(result, expected)
    
    def test_add_negative_and_positive_integer(self):
        a = -5
        b = -10
        expected = -15

        result = self.calc.add(a, b)

        self.assertEqual(result, expected)
    

if __name__ == "__main__":
    unittest.main()