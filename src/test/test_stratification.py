from __future__ import absolute_import

import unittest

from src.AgentData import *
from src.stats.diffusion_stratification import *

__author__ = 'Giulio Rossetti'
__license__ = "BSD-2-Clause"
__email__ = "giulio.rossetti@gmail.com"


class DStratTest(unittest.TestCase):

    def test_geography(self):
        activeness = SocialActiveness(filename="../../italy_data/activeness.json", gz=False)
        households = SocialContext(filename="../../italy_data/households/households_9.json.gz", gz=True)
        workplaces = SocialContext(filename="../../italy_data/private_sector/private_sector_9.json.gz", gz=True)
        workplaces.update(filename="../../italy_data/public_sector/public_sector_9.json.gz", gz=True)
        schools = SocialContext(filename="../../italy_data/schools/schools_9.json.gz", gz=True)
        schools.update(filename="../../italy_data/universities/universities_9.json.gz", gz=True)
        census = SocialContext(filename="../../italy_data/census/census_9.json.gz", gz=True)
        agents = AgentList(filename="../../italy_data/agents/agents_9.json.gz", gz=True)

        ctx = Contexts(households, census, workplaces, schools, activeness)

        st = Stratifier(agents=agents, contexts=ctx)
        st.add_iterations("phase0.json")

        res = st.geography()
        self.assertEqual(len(res), 15)
        res = st.age()
        self.assertEqual(len(res), 15)
        res = st.gender()
        self.assertEqual(len(res), 15)