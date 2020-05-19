import json
import ndlib.models.ModelConfig as mc
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from src.UTLDR import UTLDR3
from src.viz.Trends import *
from src.AgentData import *
from src.Entities import *


def get_context_agents(region: Regions):
    census = SocialContext(filename=f"italy_data/census/census_{region.value}.json.gz", gz=True)
    agents = AgentList(filename=f"italy_data/agents/agents_{region.value}.json.gz", gz=True)
    activeness = SocialActiveness(filename="italy_data/activeness.json", gz=False)
    households = SocialContext(filename=f"italy_data/households/households_{region.value}.json.gz", gz=True)
    workplaces = SocialContext(filename=f"italy_data/private_sector/private_sector_{region.value}.json.gz", gz=True)
    workplaces.update(filename=f"italy_data/public_sector/public_sector_{region.value}.json.gz", gz=True)
    schools = SocialContext(filename=f"italy_data/schools/schools_{region.value}.json.gz", gz=True)
    schools.update(filename=f"italy_data/universities/universities_{region.value}.json.gz", gz=True)
    ctx = Contexts(households, census, workplaces, schools, activeness)

    return ctx, agents


region = Regions.Toscana

ctx, agents = get_context_agents(region)

model = UTLDR3(agents=agents, contexts=ctx)
config = mc.Configuration()

config.add_model_parameter("fraction_infected", 0.000002)
config.add_model_parameter("tracing_days", 0)
config.add_model_parameter("start_day", Weekdays.Monday.value)
config.add_model_parameter("mobility", 0.05)

#### Phase 0: Before Lockdown
config.add_model_parameter("sigma", 1/4)  # incubation of 4 days
config.add_model_parameter("beta", 0.06)
config.add_model_parameter("beta_e", 0.0002)
config.add_model_parameter("gamma", 0.04)
config.add_model_parameter("omega", 0.001)

# ICU
config.add_model_parameter("icu_b", 500)
config.add_model_parameter("iota", 0.20)
# Testing exposed
config.add_model_parameter("phi_e", 0)
config.add_model_parameter("kappa_e", 0)
# Testing infected
config.add_model_parameter("phi_i", 0.1)
config.add_model_parameter("kappa_i", 0.1)

config.add_model_parameter("gamma_t", 0.08)
config.add_model_parameter("gamma_f", 0.04)
config.add_model_parameter("omega_t", 0.001)
config.add_model_parameter("omega_f", 0.0015)

model.set_initial_status(config)
iterations = model.iteration_bunch(15)
json.dump(iterations, open("phase0.json", "w"))
trends = model.build_trends(iterations)
json.dump(trends, open("trends0.json", "w"))

viz = TotalCasesTrend(model, trends)
viz.plot(filename="total_cases0.pdf")

viz = DiffusionTrend(model, trends)
viz.normalized = False
viz.plot(filename="trend0.pdf", statuses=[
    'Infected', 'Exposed', "Dead"])

viz = RtTrend(model, trends)
viz.plot(filename="RTtrend0.pdf")


#### Phase 1: Lockdown + mobility allowed at municipality level for the ones not subject to lookdown (hospitals)
model.update_model_parameter("mobility", 0.005)
model.set_mobility_limits("municipality")

# Handling lethality/recovery rates
#model.update_model_parameter("gamma_t", 0.08)
#model.update_model_parameter("gamma_f", 0.1)
#model.update_model_parameter("omega_t", 0.005)
#model.update_model_parameter("omega_f", 0.006)

# Lockdown
model.update_model_parameter("mobility", 0.008)
model.update_model_parameter("lambda", 0.99)
model.update_model_parameter("mu", 0) #1/84

model.set_lockdown(to_keep=[Ateco.Sanita.value])
iterations1 = model.iteration_bunch(84)
iterations.extend(iterations1)
json.dump(iterations, open("phase1.json", "w"))
trends = model.build_trends(iterations)
json.dump(trends, open("trends1.json", "w"))

viz = TotalCasesTrend(model, trends)
viz.plot(filename="total_cases1.pdf")

viz = DiffusionTrend(model, trends)
viz.normalized = False
viz.plot(filename="trend1.pdf", statuses=[
    'Infected', "Hospitalized_mild",
    "Hospitalized_severe_ICU", "Hospitalized_severe", "Dead"])


viz = RtTrend(model, trends)
viz.plot(filename="RTtrend1.pdf")

#### Phase 2: Partial release of lockdown (schools, research - not university), mobility allowed at provincial level
model.unset_lockdown()  # to_release=[Ateco.PA_Difesa.value]
model.set_mobility_limits("province")

iterations2 = model.iteration_bunch(30)
iterations.extend(iterations2)
json.dump(iterations, open("phase2.json", "w"))
trends = model.build_trends(iterations)
json.dump(trends, open("trends2.json", "w"))

viz = TotalCasesTrend(model, trends)
viz.plot(filename="total_cases2.pdf")

viz = DiffusionTrend(model, trends)
viz.normalized = False
viz.plot(filename="trend2.pdf", statuses=[
    'Infected', "Hospitalized_mild",
    "Hospitalized_severe_ICU", "Hospitalized_severe", "Dead"])

viz = RtTrend(model, trends)
viz.plot(filename="RTtrend2.pdf")
