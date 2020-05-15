import numpy as np
from dataclasses import dataclass
from collections import defaultdict
import json
import gzip


class SocialActiveness(object):

    def __init__(self, filename=None, gz=False):
        self.activity = {}
        if filename is not None:
            self.load(filename, gz)
        if len(self.activity) > 0:
            self.categories = set([k for _, x in self.activity.items() for _, y in x.items() for k in y])
        else:
            self.categories = []

    def load(self, filename, gz=False):
        if not gz:
            with open(filename) as f:
                self.activity = json.load(f)
        else:
            with gzip.open(filename) as f:
                self.activity = json.load(f)

    def get_value(self, agent, category='all'):
        segment = list(self.activity.keys())[0]
        if category in self.categories:
            if segment == 'age':
                return self.activity['age'][str(agent.age)][category]
            if segment == 'gender':
                return self.activity['gender'][agent.gender][category]
        else:
            return 1


class SocialContext(object):

    def __init__(self, cells=None, filename=None, gz=False):
        if cells is not None:
            self.cells = cells
        elif filename is not None:
            self.load(filename, gz)
        else:
            self.cells = {}

    def update(self, filename=None, gz=False):
        if not gz:
            with open(filename) as f:
                self.cells = json.load(f)
        else:
            with gzip.open(filename) as f:
                self.cells = dict(self.cells, **json.load(f))

    def get_sample_agents(self, cell, activity=1):
        try:
            return np.random.choice(self.cells[cell]['agents'], int(len(self.cells[cell]['agents'])*activity))
        except:
            return []

    def get_category(self, cell):
        return self.cells[cell]['category']

    def get_parent(self, cell):
        return self.cells[cell]['parent']

    def get_child(self, cell):
        return self.cells[cell]['child']

    def load(self, filename, gz=False):
        if not gz:
            with open(filename) as f:
                self.cells = json.load(f)
        else:
            with gzip.open(filename) as f:
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
                 workplaces: SocialContext = None, schools: SocialContext = None, activeness: SocialActiveness=None):

        self.activeness = activeness
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

    def get_workplace_category(self, wid):
        return self.contexts['workplaces'].get_category(wid)

    def get_school_category(self, wid):
        return self.contexts['schools'].get_category(wid)

    def get_neighbors(self, agent, restrictions=False, weekend=False, other_census=None):
        household = self.get_household(agent.household)
        if not restrictions:

            activeness = self.activeness.get_value(agent, 'census')
            if other_census is None:
                census = self.get_census_sample(agent.census, activeness)
            else:
                census = self.get_census_sample(other_census, activeness)

            work, school = [], []

            if not weekend:
                if agent.work is not None:
                    activeness = self.activeness.get_value(agent, 'work')
                    work = self.get_workplace_sample(agent.work, activeness)
                if agent.school is not None:
                    activeness = self.activeness.get_value(agent, 'school')
                    school = self.get_school_sample(agent.school, activeness)

            return list(set(household) | set(census) | set(work) | set(school))
        return list(household)


class AgentList(object):

    def __init__(self, filename=None, gz=False):

        self.population = {}
        if filename is not None:
            self.load(filename, gz)

    def number_of_nodes(self):
        return len(self.population)

    def add_agent(self, agent):
        self.population[agent.aid] = agent

    def get_agent(self, aid):
        return self.population[aid]

    def load(self, filename, gz=False):

        if not gz:
            with open(filename) as f:
                for row in f:
                    ag = json.loads(row)
                    ag = Agent(ag['aid'], ag['household'], ag['census'], ag['gender'], ag['age'],
                               ag['work'], ag['school'])
                    self.add_agent(ag)
        else:
            with gzip.open(filename) as f:
                for row in f:
                    ag = json.loads(row)
                    ag = Agent(ag['aid'], ag['household'], ag['census'], ag['gender'], ag['age'],
                               ag['work'], ag['school'])
                    self.add_agent(ag)


class ContactHistory(object):

    def __init__(self):
        self.agent_to_queue = defaultdict(list)

    def add_to_queue(self, node, queue):
        self.agent_to_queue[node].extend(queue)

    def get_contacts(self, node, iteration, delta_iteration):
        queue = self.agent_to_queue[node]
        res = [aid for aid, t in queue if delta_iteration <= iteration-t]
        return res

    def compact_queue(self, node, iteration, delta_iteration):
        self.agent_to_queue[node] = [(n, t) for n, t in self.agent_to_queue[node] if delta_iteration <= iteration-t]

    def delete(self, node):
        if node in self.agent_to_queue:
            del self.agent_to_queue[node]
        elif str(node) in self.agent_to_queue:
            del self.agent_to_queue[str(node)]

@dataclass
class Agent(object):
    aid: int
    household: str
    census: int
    gender: int = None
    age: str = None
    work: str = None
    school: str = None



