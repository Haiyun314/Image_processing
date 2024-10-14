import unittest
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from gradient import Diff

class TestDiff(unittest.TestCase):
    def setUp(self):
        # Erstelle ein Beispielbild (2D-Array) für die Tests
        self.image = np.array([[0, 0, 0, 0],
                                [0, 1, 2, 0],
                                [0, 3, 4, 0],
                                [0, 0, 0, 0]])
        self.diff = Diff()

    def test_skalarprodukt(self):
        # Teste das Skalarprodukt
        grad_im = self.diff.grad(self.image)
        lap_im = self.diff.lapl(self.image)

        grad_norm = np.sum(grad_im[0] ** 2 + grad_im[1] ** 2)
        laplace_product = -np.sum(self.image * lap_im)

        # Eine Erwartungshaltung definieren
        self.assertTrue(np.isclose(grad_norm, laplace_product),
                        "Das Skalarprodukt stimmt nicht mit dem Laplace-Produkt überein.")

if __name__ == '__main__':
    unittest.main()
