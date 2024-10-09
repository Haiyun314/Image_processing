import unittest
import numpy as np

class TestDiff(unittest.TestCase):
    def setUp(self):
        # Erstelle ein Beispielbild (2D-Array) für die Tests
        self.image = np.array([[0, 0, 0, 0],
                                [0, 1, 2, 0],
                                [0, 3, 4, 0],
                                [0, 0, 0, 0]])
        self.diff = Diff()

    def test_grad(self):
        # Berechne den Gradienten
        grad_im = self.diff.grad(self.image)

        # Erwartete Werte (unter der Annahme, dass die Implementierung korrekt ist)
        expected_grad_hori = np.array([[0, 0, 0, 0],
                                        [0, 1, 1, 0],
                                        [0, 1, 1, 0],
                                        [0, 0, 0, 0]])

        expected_grad_vert = np.array([[0, 0, 0, 0],
                                        [0, 0.5, 0.5, 0],
                                        [0, 1, 1, 0],
                                        [0, 0, 0, 0]])

        np.testing.assert_array_almost_equal(grad_im[0], expected_grad_hori)
        np.testing.assert_array_almost_equal(grad_im[1], expected_grad_vert)

    def test_lapl(self):
        # Berechne den Laplace-Operator
        lap_im = self.diff.lapl(self.image)

        # Erwartete Werte (unter der Annahme, dass die Implementierung korrekt ist)
        expected_laplace = np.array([[0, 0, 0, 0],
                                      [0, 1, 1, 0],
                                      [0, 1, 1, 0],
                                      [0, 0, 0, 0]])

        np.testing.assert_array_almost_equal(lap_im, expected_laplace)

    def test_skalarprodukt(self):
        # Teste das Skalarprodukt
        grad_im = self.diff.grad(self.image)
        lap_im = self.diff.lapl(self.image)

        grad_norm = np.sum(grad_im[0] ** 2 + grad_im[1] ** 2)
        laplace_product = -np.sum(self.image * lap_im)

        # Hier kannst du eine Erwartungshaltung definieren, z.B.:
        # np.testing.assert_almost_equal(grad_norm, laplace_product)
        # Aber zuerst musst du sicherstellen, dass du erwartete Werte berechnen kannst.
        self.assertTrue(np.isclose(grad_norm, laplace_product),
                        "Das Skalarprodukt stimmt nicht mit dem Laplace-Produkt überein.")

if __name__ == '__main__':
    unittest.main()
