import json
import ndlib.models.ModelConfig as mc
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from src.UTLDR import UTLDR3
from src.viz.Trends import *
from src.AgentData import *

region = 9  # Tuscany

census = SocialContext(filename=f"italy_data/census/census_{region}.json.gz", gz=True)
agents = AgentList(filename=f"italy_data/agents/agents_{region}.json.gz", gz=True)
activeness = SocialActiveness(filename="italy_data/activeness.json", gz=False)
households = SocialContext(filename=f"italy_data/households/households_{region}.json.gz", gz=True)

workplaces = SocialContext(filename=f"italy_data/private_sector/private_sector_{region}.json.gz", gz=True)
workplaces.update(filename=f"italy_data/public_sector/public_sector_{region}.json.gz", gz=True)

schools = SocialContext(filename=f"italy_data/schools/schools_{region}.json.gz", gz=True)
schools.update(filename=f"italy_data/universities/universities_{region}.json.gz", gz=True)

ctx = Contexts(households, census, workplaces, schools, activeness)

model = UTLDR3(agents=agents, contexts=ctx)
config = mc.Configuration()

config.add_model_parameter("fraction_infected", 0.00002)
config.add_model_parameter("tracing_days", 0)
config.add_model_parameter("start_day", 1)
config.add_model_parameter("mobility", 0.05)

#### Phase 0: Before Lockdown
config.add_model_parameter("sigma", 1/4) # incubation of 4 days
config.add_model_parameter("beta", 0.01)
config.add_model_parameter("beta_e", 0.0002)
config.add_model_parameter("gamma", 0.025)
config.add_model_parameter("omega", 0.01)

# ICU
config.add_model_parameter("icu_b", 500)
config.add_model_parameter("iota", 0.20)
# Testing exposed
config.add_model_parameter("phi_e", 0)
config.add_model_parameter("kappa_e", 0)
# Testing infected
config.add_model_parameter("phi_i", 0.1)
config.add_model_parameter("kappa_i", 0.1)

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


#### Phase 1: Lockdown + mobility allowed at municipality level for the ones not subject to lookdown (hospitals)
model.update_model_parameter("mobility", 0.005)
model.set_mobility_limits("municipality")

# Handling lethality/recovery rates
model.update_model_parameter("gamma_t", 0.08)
model.update_model_parameter("gamma_f", 0.1)
model.update_model_parameter("omega_t", 0.02)
model.update_model_parameter("omega_f", 0.03)

# Lockdown
model.update_model_parameter("mobility", 0.008)
model.update_model_parameter("lambda", 1)
model.update_model_parameter("mu", 0) #1/84

model.set_lockdown(['Q'])
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
    "Hospitalized_severe_ICU", "Hospitalized_severe",
    "Lockdown_Exposed", "Dead"])

#### Phase 2: Partial release of lockdown (schools, research - not university), mobility allowed at provincial level
model.unset_lockdown(to_release=['P', 'Q', 'school'])
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
    "Hospitalized_severe_ICU", "Hospitalized_severe",
    "Lockdown_Exposed", "Dead"])
