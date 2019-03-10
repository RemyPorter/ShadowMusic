import unittest
import numpy as np
import shaded as sp

class SquareTest(unittest.TestCase):
  def test_square_range(self):
    """Confirm that square waves fall in the correct range- -1,1."""
    sq = sp.square(sp.space(3), 100)
    self.assertEqual(np.max(sq), 1.)
    self.assertEqual(np.min(sq), -1.)

  def test_pulse(self):
    """Confirm that duty cycles work"""
    space = sp.space(3)
    fif = sp.pulse(space, 100)
    tw5 = sp.pulse(space, 100, 0.25)
    fif_high = len(np.where(fif==1.)[0])
    tw5_high = len(np.where(tw5==1.)[0])
    self.assertAlmostEqual(fif_high / len(space), 0.5, 2)
    self.assertAlmostEqual(tw5_high / len(space), 0.25, 2)

if __name__ == "__main__":
  unittest.main()