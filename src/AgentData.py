import numpy as np
from dataclasses import dataclass
import json
from netdispatch import AGraph
import networkx as nx


class SocialContext(object):

    def __init__(self, cells=None, filename=None):
        if cells is not None:
            self.cells = cells
        elif filename is not None:
            self.load(filename)
        else:
            self.cells = {}

    def get_sample_agents(self, cell, activity=1):
        return np.random.choice(self.cells[cell]['agents'],
                                int(len(self.cells[cell]['agents'])*activity))

    def get_category(self, cell):
        return self.cells[cell]['category']

    def get_parent(self, cell):
        return self.cells[cell]['parent']

    def get_child(self, cell):
        return self.cells[cell]['child']

    def load(self, filename):
        with open(filename) as f:
            self.cells = json.load(f)

    def get_contexts(self, leaf=True):
        for c in self.cells:
            if leaf:
                if self.cells[c]['child'] is None:
                    yield c
            else:
                yield c


class Contexts(object):

    def __init__(self, households: SocialContext, census: SocialContext,
                 workplaces: SocialContext = None, schools: SocialContext = None):

        self.contexts = {
            'households': households,
            'census': census,
            'workplaces': workplaces,
            'schools': schools
        }



    def get_household(self, hid):
        return self.contexts['households'].get_sample_agents(hid)

    def get_census(self, leaf=True):
        return self.contexts['census'].get_contexts(leaf)

    def get_census_sample(self, cid, activity=1):
        return self.contexts['census'].get_sample_agents(cid, activity)

    def get_school_sample(self, sid, activity=1):
        return self.contexts['schools'].get_sample_agents(sid, activity)

    def get_workplace_sample(self, wid, activity=1):
        return self.contexts['workplaces'].get_sample_agents(wid, activity)

    def get_neighbors(self, agent, restrictions=False):
        household = self.get_household(agent.household)
        if not restrictions:
            census = self.get_census_sample(agent.census, agent.activity['census'])
            work, school = [], []
            if agent.work is not None:
                work = self.get_workplace_sample(agent.work, agent.activity['work'])
            if agent.school is not None:
                school = self.get_school_sample(agent.school, agent.activity['school'])
            return list(set(household) | set(census) | set(work) | set(school))
        return list(household)


class AgentList(AGraph):

    def __init__(self, filename=None):

        super().__init__(nx.Graph())

        self.population = {}
        if filename is not None:
            self.load(filename)

    def __check_type(self):
        pass

    def nodes(self):
        pass

    def edges(self):
        pass

    def number_of_nodes(self):
        return len(self.population)

    def neighbors(self, node):
        pass

    def predecessors(self, node):
        pass

    def successors(self, node):
        pass

    def get_edge_attributes(self, attribute):
        pass

    def get_node_attributes(self, attribute):
        pass

    def add_edges(self, node, endpoints):
        pass

    def remove_edges(self, node, endpoints):
        pass

    def add_agent(self, agent):
        self.population[agent.aid] = agent

    def get_agent(self, aid):
        return self.population[aid]

    def load(self, filename):
        with open(filename) as f:
            for row in f:
                ag = json.loads(row)
                ag = Agent(ag['aid'], ag['household'], ag['census'], ag['gender'], ag['age'],
                           ag['work'], ag['school'], ag['activity'])
                self.add_agent(ag)


@dataclass
class Agent(object):
    aid: int
    household: str
    census: int
    gender: int = None
    age: str = None
    work: str = None
    school: str = None
    activity: dict = None



