from agents import GrassPatch, LandPrey, LandPredator, WaterPrey, WaterPredator, Prey, Predator, WaterPatch, VisionPatch
from model import PreyPred
from mesa.experimental.devs import ABMSimulator
from mesa.visualization import (
    CommandConsole,
    Slider,
    SolaraViz,
    SpaceRenderer,
    make_plot_component,
)
from mesa.visualization.components import AgentPortrayalStyle


def prey_pred_portrayal(agent):
    if agent is None:
        return

    portrayal = AgentPortrayalStyle(
        size=40,
        marker="o",
        zorder=2,
    )

    if isinstance(agent, LandPredator):
        portrayal.update(("color", "red"), ("zorder", 1))

    elif isinstance(agent, LandPrey):
        portrayal.update(("color", "cyan"), ("zorder", 1))

    elif isinstance(agent, WaterPredator):
        portrayal.update(("color", "orange"), ("zorder", 2))

    elif isinstance(agent, WaterPrey):
        portrayal.update(("color", "purple"), ("zorder", 2))

    elif isinstance(agent, WaterPatch):
        portrayal.update(("color", "tab:blue"), ("alpha", 0.5))
        portrayal.update(("marker", "s"), ("zorder", 1))

    elif isinstance(agent, GrassPatch):
        if agent.fully_grown:
            portrayal.update(("color", "tab:green"))
            portrayal.update(("marker", "^"), ("zorder", 0))
        else:
            portrayal.update(("color", "tab:brown"))
            portrayal.update(("marker", "s"), ("zorder", -1))

    elif isinstance(agent, VisionPatch):

        if not agent.model.show_vision:
            portrayal.update(("color", "white"), ("marker", "s"), ("alpha", 0.0), ("zorder", 0))

        elif agent.creature == Predator:
            portrayal.update(("color", "orange"), ("marker", "s"), ("alpha", 0.5), ("zorder", 2))
        else:
            portrayal.update(("color", "violet"), ("marker", "s"), ("alpha", 0.5), ("zorder", 2))


    return portrayal


model_params = {
    "seed": {
        "type": "InputText",
        "value": 62706322,
        "label": "Random Seed",
    },
    "grass": {
        "type": "Select",
        "value": True,
        "values": [True, False],
        "label": "Grass regrowth enabled?",
    },
    "show_vision": {
        "type": "Select",
        "value": False,
        "values": [True, False],
        "label": "Show vision cones?"
    },
    "initial_land_prey": Slider("Initial Land Prey Population", 90, 0, 300),
    "initial_land_pred": Slider("Initial Land Pred. Population", 15, 0, 100),
    "initial_water_prey": Slider("Initial Water Prey Population", 120, 0, 300),
    "initial_water_pred": Slider("Initial Water Pred. Population", 12, 0, 100),

    "land_prey_reproduce": Slider("Land Prey Reproduction Rate", 0.6, 0.01, 1.0, 0.01),
    "land_pred_reproduce": Slider("Land Pred. Reproduction Rate", 0.6, 0.01, 1.0,0.01,),
    "water_prey_reproduce": Slider("Water Prey Reproduction Rate", 0.4, 0.01, 1.0, 0.01),
    "water_pred_reproduce": Slider("Water Pred. Reproduction Rate", 0.45, 0.01, 1.0,0.01,),

    "land_prey_gain_from_food": Slider("Land Prey Gain From Food", 70, 0, 200),
    "land_pred_gain_from_food": Slider("Land Pred. Gain From Food", 160, 0, 200),
    "water_prey_gain_from_food": Slider("Water Prey Gain From Food", 100, 0, 200),
    "water_pred_gain_from_food": Slider("Water Pred. Gain From Food", 90, 0, 200),

    "grass_regrowth_time": Slider("Grass Regrowth Time", 40, 0, 200),
}


def post_process_space(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def post_process_lines(ax):
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))

simulator = ABMSimulator()
model = PreyPred(simulator=simulator, grass=True)

lineplot_component = make_plot_component(
    {"Land Predators": "tab:red", "Land Prey": "tab:cyan", "Water Predators": "tab:orange", "Water Prey": "tab:purple"}, #"Total Kills": "tab:red"},
    post_process=post_process_lines,
)

renderer = SpaceRenderer(
    model,
    backend="matplotlib",
)
renderer.setup_agents(prey_pred_portrayal)
renderer.draw_agents()
renderer.post_process = post_process_space

page = SolaraViz(
    model,
    renderer,
    components=[lineplot_component, CommandConsole],
    model_params=model_params,
    name="Prey Pred",
    simulator=simulator,
)
page  # noqa
