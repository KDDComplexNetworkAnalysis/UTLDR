from .DiffusionModel import DiffusionModel
from .AgentData import ContactHistory
import numpy as np
from enum import Enum


__author__ = ["Giulio Rossetti", "Letizia Milli", "Salvatore Citraro"]
__license__ = "BSD-2-Clause"


class Sociality(Enum):
    Normal = 0
    Quarantine = 1
    Lockdown = 2


class Weekdays(Enum):
    Monday = 1
    Tuesday	= 2
    Wednesday = 3
    Thursday = 4
    Friday = 5
    Saturday = 6
    Sunday = 7


class UTLDR3(DiffusionModel):

    def __init__(self, agents, contexts, seed=None):
        """

        :param agents:
        :param contexts:
        :param seed:
        """

        super(self.__class__, self).__init__(agents, contexts, seed)
        self.graph = self.agents
        self.status = {n: 0 for n in self.agents.population}
        self.c_history = ContactHistory()

        self.params['nodes']['tested'] = {n: False for n in self.agents.population}
        self.params['nodes']['ICU'] = {n: False for n in self.agents.population}
        self.params['nodes']['filtered'] = {n: Sociality.Normal for n in self.agents.population}
        self.current_active = {}
        self.icu_b = self.agents.number_of_nodes()
        self.current_day = Weekdays.Monday

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
            "Dead": 11
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
                "beta_e": {
                    "descr": "Infection rate (1/expected iterations) from exposed",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0
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
                "icu_b": {
                    "descr": "Beds availability in ICU (absolute value)",
                    "range": [0, np.infty],
                    "optional": True,
                    "default": self.agents.number_of_nodes()
                },
                "iota": {
                    "descr": "Severe case probability (needing ICU treatments)",
                    "range": [0, 1],
                    "optional": True,
                    "default": 1
                },
                "start_day": {
                    "descr": "Day of the week (1 Monday - 7 Sunday)",
                    "range": [1, 7],
                    "optional": True,
                    "default": 1
                },
                "p_mobility": {
                    "descr": "Probability of leaving the neighborhood during each iteration",
                    "range": [0, 1],
                    "optional": True,
                    "default": 0.05
                },
                "tracing_days": {
                    "descr": "Days to consider for contact tracing",
                    "range": [0, np.infty],
                    "optional": True,
                    "default": 0
                },
            },
            "nodes": dict(),
            "edges": dict(),
        }

    def iteration(self, node_status=True):
        """

        :param node_status:
        :return:
        """

        actual_status = {}
        self.current_active = {}
        self.current_day = (self.actual_iteration % 7) + 1

        if self.actual_iteration == 0:
            self.icu_b = self.params['model']['icu_b']
            self.current_day = (self.params['model']['start_day'] % len(Weekdays)) + 1

            self.active = [node for node in self.status if self.status[node] == self.available_statuses['Infected']]

            self.actual_iteration += 1
            delta, node_count, status_delta = self.status_delta(actual_status)
            if node_status:
                return {"iteration": 0, "status": actual_status.copy(),
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}
            else:
                return {"iteration": 0, "status": {},
                        "node_count": node_count.copy(), "status_delta": status_delta.copy()}

        # iterate over active agents
        for aid in self.active:

            ag = self.agents.get_agent(aid)
            u = ag.aid
            u_status = self.status[u]

            ####################### Undetected Compartment ###########################

            if u_status == self.available_statuses['Exposed']:

                if self.params['model']['beta_e'] > 0:
                    neighbors = self.__get_neighbors(ag, lockdown=False)
                    actual_status = self.__infect_neighbors(u, neighbors, actual_status, exposed=True)

                tested = np.random.random_sample()  # selection for testing
                if not self.params['nodes']['tested'][u] and tested < self.__get_threshold(ag, 'phi_e'):
                    actual_status = self.__test_exposition(ag, actual_status)
                else:
                    at = np.random.random_sample()
                    if at < self.__get_threshold(ag, 'sigma'):
                        actual_status[u] = self.available_statuses['Infected']

            elif u_status == self.available_statuses['Infected']:

                # check if the agents will infect a neighbor
                neighbors = self.__get_neighbors(ag, lockdown=False)
                actual_status = self.__infect_neighbors(u, neighbors, actual_status)

                # check testing
                tested = np.random.random_sample()  # selection for testing
                if not self.params['nodes']['tested'][u] and tested < self.__get_threshold(ag, 'phi_i'):
                    actual_status = self.__test_infection(ag, actual_status)
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

                # check if susceptible neighbors have been infected
                neighbors = self.__get_neighbors(ag, lockdown=True)
                actual_status = self.__infect_neighbors(u, neighbors, actual_status)

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
                del self.c_history[u]
#                immunity = np.random.random_sample()
#                if immunity < self.__get_threshold(ag, 's'):
#                    actual_status[u] = self.available_statuses['Susceptible']

            elif u_status == self.available_statuses['Dead']:
                del self.c_history[u]

            tst = {self.available_statuses['Susceptible']: None, self.available_statuses['Dead']: None, self.available_statuses['Recovered']: None}
            if self.status[u] not in tst:
                if u not in actual_status or actual_status[u] == self.status[u] or actual_status[u] not in tst:
                    self.current_active[u] = None

        delta, node_count, status_delta = self.status_delta(actual_status)

        for k, v in actual_status.items():
            self.status[k] = v

        self.active = self.current_active
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
        actual_status = {}

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

                if self.status[u] == self.available_statuses['Susceptible']:
                    actual_status[u] = self.available_statuses['Lockdown_Susceptible']
                    self.params['nodes']['filtered'][u] = Sociality.Lockdown

                elif self.status[u] == self.available_statuses['Exposed']:
                    actual_status[u] = self.available_statuses["Lockdown_Exposed"]
                    self.params['nodes']['filtered'][u] = Sociality.Lockdown

                elif self.status[u] == self.available_statuses['Infected']:
                    actual_status[u] = self.available_statuses['Lockdown_Infected']
                    self.params['nodes']['filtered'][u] = Sociality.Lockdown
            else:
                # node refuses lockdown
                self.params['nodes']['filtered'][u] = Sociality.Normal

        delta, node_count, status_delta = self.status_delta(actual_status)

        for k, v in actual_status.items():
            self.status[k] = v
        return {"iteration": self.actual_iteration - 1, "status": {}, "node_count": node_count.copy(),
                "status_delta": status_delta.copy()}

    def unset_lockdown(self, workplaces=None):
        """
        Remove the lockdown social limitations

        :return:
        """
        actual_status = {}

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
                if self.status[u] == self.available_statuses['Lockdown_Susceptible']:
                    actual_status[u] = self.available_statuses['Susceptible']
                elif self.status[u] == self.available_statuses['Lockdown_Exposed']:
                    actual_status[u] = self.available_statuses["Exposed"]
                elif self.status[u] == self.available_statuses['Lockdown_Infected']:
                    actual_status[u] = self.available_statuses['Infected']

        delta, node_count, status_delta = self.status_delta(actual_status)

        for k, v in actual_status.items():
            self.status[k] = v
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

    def __infect_neighbors(self, aid, neighbors, actual_status, exposed=False):
        """

        :param ag:
        :param neighbors:
        :return:
        """

        activation = "beta"
        if exposed:
            activation = "beta_e"

        infected = []

        for v in neighbors:
            bt = np.random.random_sample()
            agv = self.agents.get_agent(v)
            if bt < self.__get_threshold(agv, activation):  # identifying the proper beta for the neighbor
                actual_status[v] = self.available_statuses['Exposed']
                self.current_active[v] = None
                infected.append((v, self.actual_iteration))

        self.c_history.add_to_queue(aid, infected)

        return actual_status

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

    def __get_neighbors(self, ag, lockdown=False):
        u = ag.aid
        # identify contacts among household, neighbors and colleagues
        # individual activity levels are handled internally
        if self.params['nodes']['filtered'][u] == Sociality.Normal:
            weekend = self.current_day in [Weekdays.Saturday, Weekdays.Sunday]  # checking for work related activities
            neighbors = self.contexts.get_neighbors(ag, weekend=weekend)
        elif self.params['nodes']['filtered'][u] == Sociality.Lockdown:
            neighbors = self.contexts.get_neighbors(ag, restrictions=True)
        else:
            neighbors = []

        # filter out neighbors in quarantine or in lockdown (except household members)
        if not lockdown:

            # long range contacts due to user mobility
            long_range = self.__get_mobility(ag)
            neighbors.extend(long_range)

            neighbors = [n for n in neighbors if self.status[n] == self.available_statuses['Susceptible'] and
                         self.params['nodes']['filtered'][n] == Sociality.Normal or
                         (self.params['nodes']['filtered'][n] == Sociality.Lockdown and
                          ag.household == self.agents.get_agent(n).household)]
        else:
            neighbors = [n for n in neighbors if self.status[n] == self.available_statuses['Lockdown_Susceptible'] and
                         self.params['nodes']['filtered'][n] == Sociality.Normal or
                         (self.params['nodes']['filtered'][n] == Sociality.Lockdown and
                          ag.household == self.agents.get_agent(n).household)]

        return neighbors

    def __test_infection(self, ag, actual_status):
        u = ag.aid
        res = np.random.random_sample()  # probability of false negative result

        if res > self.__get_threshold(ag, 'kappa_i'):
            self.__limit_social_contacts(ag, 'Tested')

            # contact tracing
            if self.params['model']['tracing_days'] > 0:
                contacts = self.c_history.get_contacts(u, self.actual_iteration, self.params['model']['tracing_days'])
                actual_status = self.__contact_tracing_testing(contacts, actual_status)
                self.c_history.delete(u)

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

        return actual_status

    def __test_exposition(self, ag, actual_status):
        u = ag.aid
        res = np.random.random_sample()  # probability of false negative result
        if res > self.__get_threshold(ag, 'kappa_e'):
            self.__limit_social_contacts(ag, 'Tested')

            # contact tracing
            if self.params['model']['tracing_days'] > 0:
                contacts = self.c_history.get_contacts(u, self.actual_iteration, self.params['model']['tracing_days'])
                actual_status = self.__contact_tracing_testing(contacts, actual_status, testing=True)
                self.c_history.delete(u)

            actual_status[u] = self.available_statuses['Identified_Exposed']
        self.params['nodes']['tested'][u] = True
        return actual_status

    def __contact_tracing_testing(self, agents, actual_status, testing=False):

        target_statuses = {self.available_statuses['Susceptible']: None,
                           self.available_statuses['Lockdown_Susceptible']: None}

        for u in agents:
            if self.status[u] in target_statuses:
                ag = self.agents[u]
                res = np.random.random_sample()  # probability of false negative result

                if testing:
                    if res > self.__get_threshold(ag, 'kappa_e'):
                        self.__limit_social_contacts(ag, 'Tested')

                        actual_status[u] = self.available_statuses['Identified_Exposed']
                    self.params['nodes']['tested'][u] = True

                else:
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

        return actual_status


    def __get_mobility(self, ag):
        agent_municipality = self.contexts.contexts['census'].cells[ag.census]['parent'][0]
        agent_province = self.contexts.contexts['census'].cells[agent_municipality]['parent'][0]
        region = self.contexts.contexts['census'].cells[agent_province]['parent'][0]
        provinces = self.contexts.contexts['census'].cells[region]['child']
        provinces = list(set(provinces) - set([agent_province]))

        # @todo: tune probability from data
        p_weights = [self.params['model']['p_mobility']/(len(provinces))]*(len(provinces))
        p_weights.append(1-self.params['model']['p_mobility'])
        provinces.append(agent_province)

        selected_province = np.random.choice(provinces, 1, p=p_weights)[0]
        municipalities_selected_province = self.contexts.contexts['census'].cells[selected_province]['child']
        selected_municipality = np.random.choice(municipalities_selected_province, 1)[0]
        census_selected_municipality = self.contexts.contexts['census'].cells[selected_municipality]['child']
        selected_census = np.random.choice(census_selected_municipality, 1)[0]

        neighbors = self.contexts.get_neighbors(ag, weekend=True, other_census=selected_census)

        return neighbors
