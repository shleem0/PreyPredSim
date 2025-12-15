from agents import GrassPatch, LandPrey, LandPredator, WaterPatch, VisionPatch
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
        size=50,
        marker="o",
        zorder=2,
    )

    if isinstance(agent, LandPredator):
        portrayal.update(("color", "red"), ("zorder", 1))

    elif isinstance(agent, LandPrey):
        portrayal.update(("color", "cyan"), ("zorder", 1))

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

        elif agent.creature == LandPredator:
            portrayal.update(("color", "orange"), ("marker", "s"), ("alpha", 0.2), ("zorder", 2))
        else:
            portrayal.update(("color", "violet"), ("marker", "s"), ("alpha", 0.2), ("zorder", 2))


    return portrayal


model_params = {
    "seed": {
        "type": "InputText",
        "value": 1,
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
        "value": True,
        "values": [True, False],
        "label": "Show vision cones?"
    },
    "initial_land_prey": Slider("Initial Land Prey Population", 90, 10, 300),
    "initial_land_pred": Slider("Initial Land Pred. Population", 30, 5, 100),
    "land_prey_reproduce": Slider("Land Prey Reproduction Rate", 0.6, 0.01, 1.0, 0.01),
    "land_pred_reproduce": Slider("Land Pred. Reproduction Rate", 0.8, 0.01, 1.0,0.01,),
    "land_pred_gain_from_food": Slider("Land Pred. Gain From Food", 140, 1, 200),
    "land_prey_gain_from_food": Slider("Land Prey Gain From Food", 70, 1, 200),
    "grass_regrowth_time": Slider("Grass Regrowth Time", 50, 1, 200),
}


def post_process_space(ax):
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])


def post_process_lines(ax):
    ax.legend(loc="center left", bbox_to_anchor=(1, 0.9))


lineplot_component = make_plot_component(
    {"Land Predators": "tab:orange", "Land Prey": "tab:cyan"}, #"Total Kills": "tab:red"},
    post_process=post_process_lines,
)

simulator = ABMSimulator()
model = PreyPred(simulator=simulator, grass=True)

renderer = SpaceRenderer(
    model,
    backend="matplotlib",
)
renderer.draw_agents(prey_pred_portrayal)
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
