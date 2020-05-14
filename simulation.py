import json
import ndlib.models.ModelConfig as mc
from ndlib.viz.mpl.DiffusionTrend import DiffusionTrend
from src.UTLDR import UTLDR3
from src.viz.FatalityRateTrend import *
from src.AgentData import *

region = 9  # Tuscany

activeness = SocialActiveness(filename="italy_data/activeness.json", gz=False)
households = SocialContext(filename=f"italy_data/households/households_{region}.json.gz", gz=True)
workplaces = SocialContext(filename=f"italy_data/private_sector/private_sector_{region}.json.gz", gz=True)
schools = SocialContext(filename=f"italy_data/schools/schools_{region}.json.gz", gz=True)
census = SocialContext(filename=f"italy_data/census/census_{region}.json.gz", gz=True)
agents = AgentList(filename=f"italy_data/agents/agents_{region}.json.gz", gz=True)

ctx = Contexts(households, census, workplaces, schools, activeness)

model = UTLDR3(agents=agents, contexts=ctx)
config = mc.Configuration()

config.add_model_parameter("fraction_infected", 0.0002)
config.add_model_parameter("tracing_days", 0)
config.add_model_parameter("start_day", 1)
config.add_model_parameter("mobility", 0.05)

#### Phase 0: Before Lockdown
config.add_model_parameter("sigma", 1/4)
config.add_model_parameter("beta", 0.006)
config.add_model_parameter("beta_e", 0.0002)
config.add_model_parameter("gamma", 0.003)
config.add_model_parameter("omega", 0.0005)

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


#### Phase 1: Lockdown
model.update_model_parameter("mobility", 0.005)
# ICU
model.update_model_parameter("icu_b", 10)
model.update_model_parameter("iota", 0.20)
# Testing exposed
model.update_model_parameter("phi_e", 0)
model.update_model_parameter("kappa_e", 0)
# Testing infected
model.update_model_parameter("phi_i", 0.1)
model.update_model_parameter("kappa_i", 0.1)
# Handling lethality/recovery rates
model.update_model_parameter("gamma_t", 0.08)
model.update_model_parameter("gamma_f", 0.1)
model.update_model_parameter("omega_t", 0.01)
model.update_model_parameter("omega_f", 0.08)

# Lockdown
model.update_model_parameter("mobility", 0.008)
model.update_model_parameter("lambda", 0.8)
model.update_model_parameter("mu", 1/84)

model.set_lockdown()
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
    'Infected', 'Exposed', "Identified_Exposed", "Hospitalized_mild",
    "Hospitalized_severe_ICU", "Hospitalized_severe",
    "Lockdown_Exposed", "Dead"])

#### Phase 2: Partial release of lockdown
#model.unset_lockdown()
#iterations2 = model.iteration_bunch(10)

