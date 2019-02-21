import unittest
import numpy as np
import shaded as sp

class SquareTest(unittest.TestCase):
  def test_square_range(self):
    """Confirm that square waves fall in the correct range- -1,1."""
    sq = sp.square(sp.space(3), 100)
    self.assertEqual(np.max(sq), 1.)
    self.assertEqual(np.min(sq), -1.)

if __name__ == "__main__":
  unittest.main()