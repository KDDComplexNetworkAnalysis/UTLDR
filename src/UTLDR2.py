from ndlib.models.DiffusionModel import DiffusionModel
import numpy as np
import networkx as nx
import future
from enum import Enum

__author__ = ["Giulio Rossetti", "Letizia Milli", "Salvatore Citraro"]
__license__ = "BSD-2-Clause"


class Sociality(Enum):
    Normal = 0
    Quarantine = 1
    Lockdown = 2


class UTLDR2(DiffusionModel):

    def __init__(self, agents, contexts, seed=None):
        """

        :param agents:
        :param contexts:
        :param seed:
        """

        super(self.__class__, self).__init__(nx.Graph(), seed)

        self.agents = agents
        self.contexts = contexts

        self.status = {n: 0 for n in self.agents.population}

        self.params['nodes']['vaccinated'] = {n: False for n in self.agents.population}
        self.params['nodes']['tested'] = {n: False for n in self.agents.population}
        self.params['nodes']['ICU'] = {n: False for n in self.agents.population}
        self.params['nodes']['filtered'] = {n: Sociality.Normal for n in self.agents.population}
        self.icu_b = self.graph.number_of_nodes()

        self.name = "UTLDR"

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
            "Dead": 11,
            "Vaccinated": 12,
        }

        self.parameters = {
            "model": {
                "sigma": {
                    "descr": "Incubation rate (1/expected iterations)",
                    "range": [0, 1],
                    "optional": False
                },
                "beta": {
                    "descr": "Infection rate (1/expected iterations)",
                    "range": [0, 1],
                    "optional": False
                },
                "gamma": {
                    "descr": "Recovery rate - Mild, Asymptomatic, Paucisymptomatic (1/expected iterations)",
                    "range": [0, 1],
                    "optional": False
                },
                "gamma_t": {
                    "descr": "Recovery rate - Severe in ICU (1/expected iterations)",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0.6
                },
                "gamma_f": {
                    "descr": "Recovery rate - Severe not in ICU (1/expected iterations)",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0.95
                },
                "omega": {
                    "descr": "Death probability - Mild, Asymptomatic, Paucisymptomatic",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0
                },
                "omega_t": {
                    "descr": "Death probability - Severe in ICU",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0
                },
                "omega_f": {
                    "descr": "Death probability - Severe not in ICU",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0
                },
                "phi_e": {
                    "descr": "Testing probability if Exposed",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0
                },
                "phi_i": {
                    "descr": "Testing probability if Infected",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0
                },
                "kappa_e": {
                    "descr": "Test False Negative probability if Exposed",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0.7
                },
                "kappa_i": {
                    "descr": "Test False Negative probability if Infected",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0.9
                },
                "lambda": {
                    "descr": "Lockdown effectiveness (percentage of compliant individuals)",
                    "range": [0, 1],
                    "optional": True,
                    "default": 1
                },
                "mu": {
                    "descr": "Lockdown duration (1/expected iterations)",
                    "range": [0, 1],
                    "optional": True,
                    "default": 1
                },
                "p": {
                    "descr": "Probability of long-range interactions",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0
                },
                "p_l": {
                    "descr": "Probability of long-range interactions if in Lockdown",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0
                },
                "lsize": {
                    "descr": "Percentage of long-range interactions w.r.t short-range ones",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0.25
                },
                "icu_b": {
                    "descr": "Beds availability in ICU (absolute value)",
                    "range": [0, np.infty],
                    "optional": True,
                    "default": self.graph.number_of_nodes()
                },
                "iota": {
                    "descr": "Severe case probability (needing ICU treatments)",
                    "range": [0, 1],
                    "optional": True,
                    "default": 1
                },
                "z": {
                    "descr": "Probability of infection from corpses",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0
                },
                "s": {
                    "descr": "Probability of absent immunization",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0
                },
                "v": {
                    "descr": "Probability of vaccination (single chance per agent)",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0
                },
                "f": {
                    "descr": "Probability of Vaccination nullification (inverse of temporal coverage)",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0
                },
            },
            "nodes": dict(),
            "edges": {},
        }

    def iteration(self, node_status=True):
        """

        :param node_status:
        :return:
        """

        self.clean_initial_status(self.available_statuses.values())

        actual_status = {node: nstatus for node, nstatus in future.utils.iteritems(self.status)}

        if self.actual_iteration == 0:
            self.icu_b = self.params['model']['icu_b']

            self.actual_iteration += 1
            delta, node_count, status_delta = self.status_delta(actual_status)
            if node_status:
                return {"iteration": 0, "status": actual_status.copy(),
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}
            else:
                return {"iteration": 0, "status": {},
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}

        # iterate over census cells
        for census in self.contexts.get_census():
            population = self.contexts.get_census_sample(census)

            # iterate over the agents allocated to the current census cell
            for aid in population:
                ag = self.agents.get_agent(aid)
                u = ag.aid
                u_status = self.status[u]

                # identify contacts among household, neighbors and colleagues
                # individual activity levels are handled internally
                if self.params['nodes']['filtered'][u] == Sociality.Normal:
                    neighbors = self.contexts.get_neighbors(ag)
                elif self.params['nodes']['filtered'][u] == Sociality.Lockdown:
                    neighbors = self.contexts.get_neighbors(ag, restrictions=True)
                else:
                    neighbors = []

                # filter out neighbors in quarantine or in lockdown (except household members)
                neighbors = [n for n in neighbors if self.params['nodes']['filtered'][n] == Sociality.Normal or
                             (self.params['nodes']['filtered'][
                                  n] == Sociality.Lockdown and ag.household == self.agents.get_agent(n).household)]

                ####################### Undetected Compartment ###########################

                if u_status == self.available_statuses['Susceptible']:
                    actual_status[u] = self.__Susceptible_to_Exposed(ag, neighbors, lockdown=False)

                elif u_status == self.available_statuses['Exposed']:

                    tested = np.random.random_sample()  # selection for testing
                    if not self.params['nodes']['tested'][u] and tested < self.__get_threshold(ag, 'phi_e'):
                        res = np.random.random_sample()  # probability of false negative result
                        if res > self.__get_threshold(ag, 'kappa_e'):
                            self.__limit_social_contacts(ag, 'Tested')
                            actual_status[u] = self.available_statuses['Identified_Exposed']
                        self.params['nodes']['tested'][u] = True
                    else:
                        at = np.random.random_sample()
                        if at < self.__get_threshold(ag, 'sigma'):
                            actual_status[u] = self.available_statuses['Infected']

                elif u_status == self.available_statuses['Infected']:

                    tested = np.random.random_sample()  # selection for testing
                    if not self.params['nodes']['tested'][u] and tested < self.__get_threshold(ag, 'phi_i'):
                        res = np.random.random_sample()  # probability of false negative result
                        if res > self.__get_threshold(ag, 'kappa_i'):
                            self.__limit_social_contacts(ag, 'Tested')

                            icup = np.random.random_sample()  # probability of severe case needing ICU

                            if icup < self.__get_threshold(ag, 'iota'):
                                if self.icu_b > 0 and not self.params['nodes']['ICU'][u]:
                                    actual_status[u] = self.available_statuses['Hospitalized_severe_ICU']
                                    self.icu_b -= 1
                                else:
                                    actual_status[u] = self.available_statuses['Hospitalized_severe']
                            else:
                                actual_status[u] = self.available_statuses['Hospitalized_mild']
                        self.params['nodes']['tested'][u] = True

                    else:
                        recovered = np.random.random_sample()
                        if recovered < self.__get_threshold(ag, 'gamma'):
                            actual_status[u] = self.available_statuses['Recovered']
                        else:
                            dead = np.random.random_sample()
                            if dead < self.__get_threshold(ag, 'omega'):
                                actual_status[u] = self.available_statuses['Dead']

                ####################### Quarantined Compartments ###########################

                elif u_status == self.available_statuses['Identified_Exposed']:
                    at = np.random.random_sample()
                    if at < self.__get_threshold(ag, 'sigma'):
                        icup = np.random.random_sample()

                        if icup < self.__get_threshold(ag, 'iota'):
                            if self.icu_b > 0 and not self.params['nodes']['ICU'][u]:
                                actual_status[u] = self.available_statuses['Hospitalized_severe_ICU']
                                self.icu_b -= 1
                            else:
                                actual_status[u] = self.available_statuses['Hospitalized_severe']
                        else:
                            actual_status[u] = self.available_statuses['Hospitalized_mild']
                        self.params['nodes']['ICU'][u] = True

                elif u_status == self.available_statuses['Hospitalized_mild']:
                    recovered = np.random.random_sample()
                    if recovered < self.__get_threshold(ag, 'gamma'):
                        actual_status[u] = self.available_statuses['Recovered']
                    else:
                        dead = np.random.random_sample()
                        if dead < self.__get_threshold(ag, 'omega'):
                            actual_status[u] = self.available_statuses['Dead']

                elif u_status == self.available_statuses['Hospitalized_severe']:
                    recovered = np.random.random_sample()
                    if recovered < self.__get_threshold(ag, 'gamma_f'):
                        actual_status[u] = self.available_statuses['Recovered']
                    else:
                        dead = np.random.random_sample()
                        if dead < self.__get_threshold(ag, 'omega_f'):
                            actual_status[u] = self.available_statuses['Dead']

                elif u_status == self.available_statuses['Hospitalized_severe_ICU']:
                    recovered = np.random.random_sample()
                    if recovered < self.__get_threshold(ag, 'gamma_t'):
                        actual_status[u] = self.available_statuses['Recovered']
                        self.icu_b += 1
                    else:
                        dead = np.random.random_sample()
                        if dead < self.__get_threshold(ag, 'omega_t'):
                            actual_status[u] = self.available_statuses['Dead']
                            self.icu_b += 1

                ####################### Lockdown Compartments ###########################

                elif u_status == self.available_statuses['Lockdown_Susceptible']:
                    # test lockdown exit
                    exit_flag = np.random.random_sample()  # loockdown acceptance
                    if exit_flag < self.__get_threshold(ag, 'mu'):
                        actual_status[u] = self.available_statuses['Susceptible']
                        self.__ripristinate_social_contacts(u)

                    else:
                        actual_status[u] = self.__Susceptible_to_Exposed(ag, neighbors, lockdown=True)

                elif u_status == self.available_statuses['Lockdown_Exposed']:
                    # test lockdown exit
                    exit_flag = np.random.random_sample()  # loockdown exit
                    if exit_flag < self.__get_threshold(ag, 'mu'):
                        actual_status[u] = self.available_statuses['Exposed']
                        self.__ripristinate_social_contacts(u)

                    else:
                        at = np.random.random_sample()
                        if at < self.__get_threshold(ag, 'sigma'):
                            actual_status[u] = self.available_statuses['Lockdown_Infected']

                elif u_status == self.available_statuses['Lockdown_Infected']:
                    # test lockdown exit
                    exit_flag = np.random.random_sample()

                    if exit_flag < self.__get_threshold(ag, 'mu'):
                        actual_status[0] = self.available_statuses['Infected']
                        self.__ripristinate_social_contacts(u)

                    else:
                        dead = np.random.random_sample()
                        if dead < self.__get_threshold(ag, 'omega'):
                            actual_status[u] = self.available_statuses['Dead']
                        else:
                            recovered = np.random.random_sample()
                            if recovered < self.__get_threshold(ag, 'gamma'):
                                actual_status[u] = self.available_statuses['Recovered']

                ####################### Resolved Compartments ###########################

                elif u_status == self.available_statuses['Recovered']:
                    immunity = np.random.random_sample()
                    if immunity < self.__get_threshold(ag, 's'):
                        actual_status[u] = self.available_statuses['Susceptible']

                elif u_status == self.available_statuses['Vaccinated']:
                    failure = np.random.random_sample()
                    if failure < self.__get_threshold(ag, 'f'):
                        self.params['nodes']['vaccinated'][u] = False
                        actual_status[u] = self.available_statuses['Susceptible']

                elif u_status == self.available_statuses['Dead']:
                    pass

        delta, node_count, status_delta = self.status_delta(actual_status)
        self.status = actual_status
        self.actual_iteration += 1

        if node_status:
            return {"iteration": self.actual_iteration - 1, "status": delta.copy(),
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}
        else:
            return {"iteration": self.actual_iteration - 1, "status": {},
                    "node_count": node_count.copy(), "status_delta": status_delta.copy()}

    ###################################################################################################################

    def add_ICU_beds(self, n):
        """
        Add/Subtract beds in intensive care

        :param n: number of beds to add/remove
        :return:
        """
        self.icu_b = max(0, self.icu_b + n)

    def set_lockdown(self, workplaces=None):
        """
        Impose the beginning of a lockdown

        :param workplaces: (optional) list of workplaces/school ids to close.
        :return:
        """
        actual_status = {node: nstatus for node, nstatus in future.utils.iteritems(self.status)}

        for u, ag in self.agents.population.items():

            candidate = True
            if workplaces is not None and 'work' in self.params['nodes']:
                if len(set(workplaces) & set(self.params['nodes']['work'][u])) == 0:
                    candidate = False

            if not candidate:
                continue

            # loockdown acceptance
            la = np.random.random_sample()
            if la < self.__get_threshold(ag, 'lambda'):

                if actual_status[u] == self.available_statuses['Susceptible']:
                    actual_status[u] = self.available_statuses['Lockdown_Susceptible']
                    self.params['nodes']['filtered'][u] = Sociality.Lockdown

                elif actual_status[u] == self.available_statuses['Exposed']:
                    actual_status[u] = self.available_statuses["Lockdown_Exposed"]
                    self.params['nodes']['filtered'][u] = Sociality.Lockdown

                elif actual_status[u] == self.available_statuses['Infected']:
                    actual_status[u] = self.available_statuses['Lockdown_Infected']
                    self.params['nodes']['filtered'][u] = Sociality.Lockdown
            else:
                # node refuses lockdown
                self.params['nodes']['filtered'][u] = Sociality.Normal

        delta, node_count, status_delta = self.status_delta(actual_status)
        self.status = actual_status
        return {"iteration": self.actual_iteration - 1, "status": {}, "node_count": node_count.copy(),
                "status_delta": status_delta.copy()}

    def unset_lockdown(self, workplaces=None):
        """
        Remove the lockdown social limitations

        :return:
        """
        actual_status = {node: nstatus for node, nstatus in future.utils.iteritems(self.status)}

        for _, ag in self.agents.population.items():
            u = ag.aid
            flag = False
            if workplaces is None:
                self.__ripristinate_social_contacts(u)
                flag = True
            else:
                for w in workplaces:
                    if (ag.work is not None and w in ag.work) or (ag.school is not None and w in ag.school):
                        self.__ripristinate_social_contacts(u)
                        flag = True
                        break

            if flag:
                if actual_status[u] == self.available_statuses['Lockdown_Susceptible']:
                    actual_status[u] = self.available_statuses['Susceptible']
                elif actual_status[u] == self.available_statuses['Lockdown_Exposed']:
                    actual_status[u] = self.available_statuses["Exposed"]
                elif actual_status[u] == self.available_statuses['Lockdown_Infected']:
                    actual_status[u] = self.available_statuses['Infected']

        delta, node_count, status_delta = self.status_delta(actual_status)
        self.status = actual_status
        return {"iteration": self.actual_iteration + 1, "status": {}, "node_count": node_count.copy(),
                "status_delta": status_delta.copy()}

    def __limit_social_contacts(self, ag, event='Tested'):
        """

        :param ag:
        :param event:
        :return:
        """
        u = ag.aid

        # Quarantine
        if event == 'Tested':
            self.params['nodes']['filtered'][u] = Sociality.Quarantine

        # Lockdown (limit to households)
        else:
            self.params['nodes']['filtered'][u] = Sociality.Lockdown

    def __ripristinate_social_contacts(self, u):
        self.params['nodes']['filtered'][u] = Sociality.Normal

    ####################### Undetected Compartment ###########################

    def __Susceptible_to_Exposed(self, ag, neighbors, lockdown=False):
        """

        :param ag:
        :param neighbors:
        :param lockdown:
        :return:
        """
        u = ag.aid
        # vaccination test
        if self.__get_threshold(ag, 'v') > 0 and not self.params['nodes']['vaccinated'][u]:
            v_prob = np.random.random_sample()
            if v_prob < self.__get_threshold(ag, 'v'):
                self.params['nodes']['vaccinated'][u] = True
                return self.available_statuses['Vaccinated']

        social_interactions = len(neighbors)

        l_range_proba = self.__get_threshold(ag, 'p')
        if lockdown:
            l_range_proba = self.__get_threshold(ag, 'p_l')

        l_range = np.random.random_sample()  # long range interaction
        if l_range < l_range_proba:
            # filtering out quarantined and dead nodes
            if self.__get_threshold(ag, 'z') == 0:
                candidates = [n for n in self.agents.population if self.status[n] not in
                              [self.available_statuses['Identified_Exposed'],
                               self.available_statuses['Hospitalized_mild'],
                               self.available_statuses['Dead']]]
            else:
                candidates = [n for n in self.agents.population if self.status[n] not in
                              [self.available_statuses['Identified_Exposed'],
                               self.available_statuses['Hospitalized_mild']]
                              ]

            neighbors.extend(list(np.random.choice(a=candidates,
                                                   size=int(social_interactions * self.params['model']['lsize']),
                                                   replace=True)))

        for v in neighbors:
            if self.status[v] == self.available_statuses['Infected'] or \
                    self.status[v] == self.available_statuses['Lockdown_Infected'] or \
                    self.status[v] == self.available_statuses['Hospitalized_mild']:
                bt = np.random.random_sample()

                if bt < self.__get_threshold(ag, 'beta'):
                    if lockdown:
                        return self.available_statuses['Lockdown_Exposed']
                    return self.available_statuses['Exposed']

            elif self.status[v] == self.available_statuses['Dead']:
                zp = np.random.random_sample()
                if zp < self.__get_threshold(ag, 'z'):  # infection risk due to partial corpse disposal
                    if lockdown:
                        return self.available_statuses['Lockdown_Exposed']
                    return self.available_statuses['Exposed']

        if lockdown:
            return self.available_statuses['Lockdown_Susceptible']
        return self.available_statuses['Susceptible']

    def __get_threshold(self, ag, parameter):
        """

        :param ag:
        :param parameter:
        :return:
        """
        # stratified population scenario
        if isinstance(self.params['model'][parameter], dict):

            if ag.gender in self.params['model'][parameter]:
                nclass = ag.gender
            elif ag.age in self.params['model'][parameter]:
                nclass = ag.age
            else:
                raise ValueError(f"Parameter {parameter} not specified for {ag}")

            return self.params['model'][parameter][nclass]
        # base scenario, single value
        else:
            return self.params['model'][parameter]
