import unittest
import numpy as np
import shaded as sp

class ArpTest(unittest.TestCase):
  def setUp(self):
    self.steps = [0,1,2,3]

  def test_arp(self):
    space = np.linspace(0,10)
    arped = sp.arp(space, [1,2,3])
    self.assertEqual(arped[0], 1.)
    self.assertEqual(arped[17], 2.)

if __name__ == "__main__":
  unittest.main()