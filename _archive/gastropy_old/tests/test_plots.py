# Author: Nicolas Legrand <nicolas.legrand@cfin.au.dk>

import unittest
from unittest import TestCase

import numpy as np

from gastropy.plots import plot_raw


class TestHrv(TestCase):
    def test_plot_raw(self):
        """Test plot_raw function"""
        signal1, signal2 = np.random.normal(0, 1, 10000), np.random.normal(0, 1, 10000)
        plot_raw(signal=[signal1, signal2])


if __name__ == "__main__":
    unittest.main(argv=["first-arg-is-ignored"], exit=False)
