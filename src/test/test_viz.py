from __future__ import absolute_import

import unittest
import json
from src.viz.Trends import TotalCasesTrend

__author__ = 'Giulio Rossetti'
__license__ = "BSD-2-Clause"
__email__ = "giulio.rossetti@gmail.com"


class VizTest(unittest.TestCase):

    def test_total(self):
        res = json.load(open("phase0.json"))
        pass
