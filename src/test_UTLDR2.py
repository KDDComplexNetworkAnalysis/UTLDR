from __future__ import absolute_import

import unittest

import ndlib.models.ModelConfig as mc
from .UTLDR2 import UTLDR2
from .AgentData import *

__author__ = 'Giulio Rossetti'
__license__ = "BSD-2-Clause"
__email__ = "giulio.rossetti@gmail.com"


class UTLDRTest(unittest.TestCase):

    def test_utldr(self):
        households = SocialContext(filename="data_sample/households.json")
        workplaces = SocialContext(filename="data_sample/workplaces.json")
        schools = SocialContext(filename="data_sample/schools.json")
        census = SocialContext(filename="data_sample/census.json")
        agents = AgentList(filename="data_sample/agents.json")

        ctx = Contexts(households, census, workplaces, schools)

        model = UTLDR2(agents=agents, contexts=ctx)
        config = mc.Configuration()

        # Undetected
        config.add_model_parameter("sigma", 0.05)
        config.add_model_parameter("beta", {"M": 0.4, "F": 0})
        config.add_model_parameter("gamma", 0.05)
        config.add_model_parameter("omega", 0.01)
        config.add_model_parameter("p", 0.04)
        config.add_model_parameter("lsize", 0.2)

        # Testing
        config.add_model_parameter("phi_e", 0.03)
        config.add_model_parameter("phi_i", 0.1)
        config.add_model_parameter("kappa_e", 0.03)
        config.add_model_parameter("kappa_i", 0.1)
        config.add_model_parameter("gamma_t", 0.08)
        config.add_model_parameter("gamma_f", 0.1)
        config.add_model_parameter("omega_t", 0.01)
        config.add_model_parameter("omega_f", 0.08)
        config.add_model_parameter("icu_b", 10)
        config.add_model_parameter("iota", 0.20)
        config.add_model_parameter("z", 0.2)
        config.add_model_parameter("s", 0.05)

        # Lockdown
        config.add_model_parameter("lambda", 0.8)
        config.add_model_parameter("mu", 0.05)
        config.add_model_parameter("p_l", 0.04)

        # Vaccination
        config.add_model_parameter("v", 0.15)
        config.add_model_parameter("f", 0.02)

        model.set_initial_status(config)
        iterations = model.iteration_bunch(10)
        self.assertEqual(len(iterations), 10)

        model.set_lockdown()
        iterations = model.iteration_bunch(10)
        self.assertEqual(len(iterations), 10)
        iterations = model.iteration_bunch(10, node_status=False)
        self.assertEqual(len(iterations), 10)

        model.unset_lockdown()
        iterations = model.iteration_bunch(10)
        self.assertEqual(len(iterations), 10)
        iterations = model.iteration_bunch(10, node_status=False)
        self.assertEqual(len(iterations), 10)

        model.set_lockdown(['W1_1'])
        iterations = model.iteration_bunch(10)
        self.assertEqual(len(iterations), 10)
        iterations = model.iteration_bunch(10, node_status=False)
        self.assertEqual(len(iterations), 10)

        model.unset_lockdown(['W1_1', 'W1_2'])
        iterations = model.iteration_bunch(10)
        self.assertEqual(len(iterations), 10)
        iterations = model.iteration_bunch(10, node_status=False)
        self.assertEqual(len(iterations), 10)


class AgentDataTest(unittest.TestCase):

    def test_SocialContext(self):
        sc = SocialContext()
        sc.load("data_sample/workplaces.json")
        for c in sc.get_contexts():
            self.assertIsInstance(sc.get_sample_agents(c, 0.6), np.ndarray)
            self.assertIsInstance(sc.get_category(c), str)
            child = sc.get_child(c)
            partent = sc.get_parent(c)
            if child is not None:
                self.assertIsInstance(child, str)
            if partent is not None:
                self.assertIsInstance(partent, str)

    def test_contexts(self):
        households = SocialContext(filename="data_sample/households.json")
        workplaces = SocialContext(filename="data_sample/workplaces.json")
        schools = SocialContext(filename="data_sample/schools.json")
        census = SocialContext(filename="data_sample/census.json")
        ctx = Contexts(households, census, workplaces, schools)
        self.assertIsInstance(ctx.get_household("H1"), np.ndarray)
        self.assertIsInstance(ctx.get_census_sample("C1", activity=1), np.ndarray)
        self.assertIsInstance(ctx.get_school_sample("S1", activity=1), np.ndarray)
        self.assertIsInstance(ctx.get_workplace_sample("W1", activity=1), np.ndarray)

    def test_agents(self):
        households = SocialContext(filename="data_sample/households.json")
        workplaces = SocialContext(filename="data_sample/workplaces.json")
        schools = SocialContext(filename="data_sample/schools.json")
        census = SocialContext(filename="data_sample/census.json")
        agents = AgentList(filename="data_sample/agents.json")

        ctx = Contexts(households, census, workplaces, schools)
        for census in ctx.get_census():
            population = ctx.get_census_sample(census)
            for aid in population:
                ag = agents.get_agent(aid)
                self.assertIsInstance(ag, Agent)
                self.assertIsInstance(ctx.get_neighbors(ag), list)
