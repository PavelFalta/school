from TSW.zadani1.calculator import Calculator
import unittest


class TestCalculator(unittest.TestCase): 
    def setUp(self):
        self.calc = Calculator()

    def test_add(self): 
        self.assertEqual(self.calc.add(2, 3), 5) 
        self.assertEqual(self.calc.add(-1, 1), 0)

    def test_subtract(self):
        self.assertEqual(self.calc.subtract(5, 2), 3)
        self.assertEqual(self.calc.subtract(2, 5), -3)

    def test_multiply(self):
        self.assertEqual(self.calc.multiply(3, 4), 12)
        self.assertEqual(self.calc.multiply(0, 5), 0)

    def test_divide(self):
        self.assertEqual(self.calc.divide(10, 2), 5)
        self.assertRaises(ValueError, self.calc.divide, 10, 0)

if __name__ == "__main__":
    unittest.main()