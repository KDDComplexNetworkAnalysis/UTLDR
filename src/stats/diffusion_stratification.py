import json
from collections import defaultdict
import tqdm


class Stratifier(object):

    def __init__(self, agents,  contexts, iterations=None):

        self.iterations = iterations
        if iterations is None:
            self.iterations = []
        self.agents = agents
        self.contexts = contexts

        self.available_statuses = {
            "Susceptible": 0,
            "Exposed": 2,
            "Infected": 1,
            "Recovered": 3,
            "Identified_Exposed": 4,
            "Hospitalized_mild": 5,
            "Hospitalized_severe_ICU": 6,
            "Hospitalized_severe": 7,
            "Lockdown_Susceptible": 8,
            "Lockdown_Exposed": 9,
            "Lockdown_Infected": 10,
            "Dead": 11
        }

    def add_iterations(self, filename):
        its = json.load(open(filename))
        self.iterations.extend(its)

    def geography(self, statuses=['Infected']):

        results = []

        statuses = {self.available_statuses[x]: None for x in statuses}

        for i in tqdm.tqdm(self.iterations):

            iid = i['iteration']
            nodes = i['status']

            stratification = {
                'province': defaultdict(int),
                'municipality': defaultdict(int),
                'census': defaultdict(int)
            }

            for aid in nodes:
                if nodes[aid] in statuses:
                    aid = int(aid)
                    ag = self.agents.get_agent(aid)
                    agent_municipality = self.contexts.contexts['census'].cells[str(ag.census)]['parent'][0]
                    agent_province = self.contexts.contexts['census'].cells[str(agent_municipality)]['parent'][0]

                    stratification['province'][agent_province] += 1
                    stratification['municipality'][agent_municipality] += 1
                    stratification['census'][ag.census] += 1

            results.append({'iteration': iid, 'stratification': stratification})

        return results

    def gender(self, statuses=['Infected']):

        results = []

        statuses = {self.available_statuses[x]: None for x in statuses}

        for i in tqdm.tqdm(self.iterations):

            iid = i['iteration']
            nodes = i['status']

            stratification = defaultdict(int)

            for aid in nodes:
                if nodes[aid] in statuses:
                    aid = int(aid)
                    ag = self.agents.get_agent(aid)
                    stratification[ag.gender] += 1

            results.append({'iteration': iid, 'stratification': stratification})

        return results

    def age(self, statuses=['Infected']):

        results = []

        statuses = {self.available_statuses[x]: None for x in statuses}

        for i in tqdm.tqdm(self.iterations):

            iid = i['iteration']
            nodes = i['status']

            stratification = defaultdict(int)

            for aid in nodes:
                if nodes[aid] in statuses:
                    aid = int(aid)
                    ag = self.agents.get_agent(aid)
                    age = ag.age
                    stratification[str(age)] += 1

            results.append({'iteration': iid, 'stratification': stratification})

        return results
