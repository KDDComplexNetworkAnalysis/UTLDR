from ndlib.viz.mpl.DiffusionViz import DiffusionPlot
import numpy as np
from copy import copy
import matplotlib as mpl
import os
if os.environ.get('DISPLAY', '') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt

__author__ = 'Giulio Rossetti'
__license__ = "BSD-2-Clause"
__email__ = "giulio.rossetti@gmail.com"


class FatalityRateTrend(DiffusionPlot):

    def __init__(self, model, trends):
        """
        :param model: The model object
        :param trends: The computed simulation trends
        """
        super(self.__class__, self).__init__(model, trends)
        self.ylabel = "#Nodes"
        self.title = "R's Trend"

    def iteration_series(self):

        dead = self.trends[0]['trends']['node_count'][self.model.available_statuses['Dead']]

        infected = self.trends[0]['trends']['node_count'][self.model.available_statuses['Infected']]
        exposed = self.trends[0]['trends']['node_count'][self.model.available_statuses['Exposed']]
        infected_lokdown = self.trends[0]['trends']['node_count'][self.model.available_statuses['Lockdown_Infected']]
        exposed_lockdown = self.trends[0]['trends']['node_count'][self.model.available_statuses['Lockdown_Exposed']]

        hospitalized_mild = self.trends[0]['trends']['node_count'][self.model.available_statuses['Hospitalized_mild']]
        hospitalized_icu = self.trends[0]['trends']['node_count'][self.model.available_statuses['Hospitalized_severe_ICU']]
        hospitalized_severe = self.trends[0]['trends']['node_count'][self.model.available_statuses['Hospitalized_severe']]
        identified_exposed = self.trends[0]['trends']['node_count'][self.model.available_statuses['Identified_Exposed']]

        for i in [exposed, infected_lokdown, exposed_lockdown, hospitalized_icu, hospitalized_mild,hospitalized_severe, identified_exposed]:
            infected = np.add(infected, i)

        infected = [x if x < np.infty else 0 for x in infected]

        ifr = np.divide(dead, infected)

        identified = copy(hospitalized_mild)
        for i in [hospitalized_icu, hospitalized_severe, identified_exposed]:
            identified = np.add(identified, i)

        identified = [x if x < np.infty else 0 for x in identified]

        cfr = np.divide(dead, identified)

        series = {"IFR": ifr, "CFR": cfr}

        return series

    def plot(self, filename=None):
        """
        Generates the plot

        :param filename: Output filename
        """

        pres = self.iteration_series()
        plt.figure(figsize=(20, 10))
        mx = 0
        for k, l in pres.items():
            mx = len(l)
            plt.plot(list(range(0, mx)), l, lw=2, label=k, alpha=0.5)

        plt.grid(axis="y")
        plt.xlabel("Iterations", fontsize=24)
        plt.ylabel(self.ylabel, fontsize=24)
        plt.legend(loc="best", fontsize=18)
        plt.xlim((0, mx))

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
            plt.clf()
        else:
            plt.show()


class R0Trend(DiffusionPlot):

    # e (tc) "totale_casi" con x=7 secondo la formula: R-n = (tc - tc-x) / (tc-x - tc-2x)

    def __init__(self, model, trends, displacement=7):
        """
        :param model: The model object
        :param trends: The computed simulation trends
        """
        super(self.__class__, self).__init__(model, trends)
        self.ylabel = "#Nodes"
        self.title = "R's Trend"
        self.displacement = displacement

    def iteration_series(self):

        infected = self.trends[0]['trends']['node_count'][self.model.available_statuses['Infected']]
        exposed = self.trends[0]['trends']['node_count'][self.model.available_statuses['Exposed']]
        infected_lokdown = self.trends[0]['trends']['node_count'][self.model.available_statuses['Lockdown_Infected']]
        exposed_lockdown = self.trends[0]['trends']['node_count'][self.model.available_statuses['Lockdown_Exposed']]
        hospitalized_mild = self.trends[0]['trends']['node_count'][self.model.available_statuses['Hospitalized_mild']]
        hospitalized_icu = self.trends[0]['trends']['node_count'][self.model.available_statuses['Hospitalized_severe_ICU']]
        hospitalized_severe = self.trends[0]['trends']['node_count'][self.model.available_statuses['Hospitalized_severe']]
        identified_exposed = self.trends[0]['trends']['node_count'][self.model.available_statuses['Identified_Exposed']]

        for i in [exposed, infected_lokdown, exposed_lockdown, hospitalized_icu, hospitalized_mild,hospitalized_severe, identified_exposed]:
            infected = np.add(infected, i)

        infected = [x if x < np.infty else 0 for x in infected]
        #R - n = (tc - tc - x) / (tc - x - tc - 2x)
        Rnr = [(x-infected[i-self.displacement])/(infected[i-self.displacement]-infected[i-(2*self.displacement)]) if
               i-(2*self.displacement)>0 and
                    infected[i - self.displacement] - infected[i - (2 * self.displacement)] > 0
               else 0 for i, x in enumerate(infected)]


        identified = copy(hospitalized_mild)
        for i in [hospitalized_icu, hospitalized_severe, identified_exposed]:
            identified = np.add(identified, i)

        identified = [x if x < np.infty else 0 for x in identified]
        Rn = [(x - identified[i - self.displacement]) / (
                    identified[i - self.displacement] - identified[i - (2 * self.displacement)]) if i - (
                    2 * self.displacement) > 0 and
                    identified[i - self.displacement] - identified[i - (2 * self.displacement)] > 0
              else 0 for i, x in enumerate(identified)]

        series = {"R0 observed": Rn, "R0 real": Rnr}

        return series

    def plot(self, filename=None):
        """
        Generates the plot

        :param filename: Output filename
        """

        pres = self.iteration_series()
        plt.figure(figsize=(20, 10))
        mx = 0
        for k, l in pres.items():
            mx = len(l)
            plt.plot(list(range(0, mx)), l, lw=2, label=k, alpha=0.5)

        plt.grid(axis="y")
        plt.xlabel("Iterations", fontsize=24)
        plt.ylabel(self.ylabel, fontsize=24)
        plt.legend(loc="best", fontsize=18)
        plt.xlim((0, mx))

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
            plt.clf()
        else:
            plt.show()


class TotalCasesTrend(DiffusionPlot):

    def __init__(self, model, trends):
        """
        :param model: The model object
        :param trends: The computed simulation trends
        """
        super(self.__class__, self).__init__(model, trends)
        self.ylabel = "#Nodes"
        self.title = "Total Cases"

    def iteration_series(self):

        dead = self.trends[0]['trends']['node_count'][self.model.available_statuses['Dead']]
        recovered = self.trends[0]['trends']['node_count'][self.model.available_statuses['Recovered']]
        infected = self.trends[0]['trends']['node_count'][self.model.available_statuses['Infected']]
        exposed = self.trends[0]['trends']['node_count'][self.model.available_statuses['Exposed']]
        infected_lokdown = self.trends[0]['trends']['node_count'][self.model.available_statuses['Lockdown_Infected']]
        exposed_lockdown = self.trends[0]['trends']['node_count'][self.model.available_statuses['Lockdown_Exposed']]
        hospitalized_mild = self.trends[0]['trends']['node_count'][self.model.available_statuses['Hospitalized_mild']]
        hospitalized_icu = self.trends[0]['trends']['node_count'][self.model.available_statuses['Hospitalized_severe_ICU']]
        hospitalized_severe = self.trends[0]['trends']['node_count'][self.model.available_statuses['Hospitalized_severe']]
        identified_exposed = self.trends[0]['trends']['node_count'][self.model.available_statuses['Identified_Exposed']]
        identified_cases = self.trends[0]['trends']['identified_cases']

        for i in [dead, recovered, exposed, infected_lokdown, exposed_lockdown, hospitalized_icu, hospitalized_mild,hospitalized_severe, identified_exposed]:
            infected = np.add(infected, i)

        series = {"Total": infected, "Identified": identified_cases}

        return series

    def plot(self, filename=None):
        """
        Generates the plot

        :param filename: Output filename
        """

        pres = self.iteration_series()
        plt.figure(figsize=(20, 10))
        mx = 0
        for k, l in pres.items():
            mx = len(l)
            plt.plot(list(range(0, mx)), l, lw=2, label=k, alpha=0.5)

        plt.grid(axis="y")
        plt.xlabel("Iterations", fontsize=24)
        plt.ylabel(self.ylabel, fontsize=24)
        plt.legend(loc="best", fontsize=18)
        plt.xlim((0, mx))

        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename)
            plt.clf()
        else:
            plt.show()

