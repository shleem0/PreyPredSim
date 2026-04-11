import math

import numpy as np

from constants import MATURITY

import matplotlib.pyplot as plt

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid
from agents import GrassPatch, WaterPatch, VisionPatch, LandPrey, LandPredator, WaterPrey, WaterPredator, Prey, Predator
from mesa.experimental.devs import ABMSimulator

from perlin_noise import PerlinNoise

import os


class PreyPred(Model):
    """Prey/Predator Model.

    A model for simulating predator-prey ecosystem modelling.
    """

    description = (
        "A model for simulating predator-prey ecosystem modelling."
    )

    def __init__(
        self,
        x=50,
        y=50,

        initial_land_prey=90,
        land_prey_max_eng=200,
        land_prey_max_hydration=200,
        land_prey_reproduce=0.6,
        land_prey_gain_from_food=70,
        land_prey_gain_from_water=100,

        initial_land_pred=15,
        land_pred_max_eng=200,
        land_pred_max_hydration=200,
        land_pred_reproduce=0.6,
        land_pred_gain_from_food=160,
        land_pred_gain_from_water=100,
        
        initial_water_prey=120,
        water_prey_max_eng=200,
        water_prey_max_hydration=200,
        water_prey_reproduce=0.4,
        water_prey_gain_from_food=100,

        initial_water_pred=12,
        water_pred_max_eng=200,
        water_pred_max_hydration=200,
        water_pred_reproduce=0.45,
        water_pred_gain_from_food=90,
        
        grass = True,
        grass_regrowth_time=40,
        lakes = True,
        total_kills = 0,

        prey_vis_r=10,
        prey_vis_a=180,

        land_pred_vis_r=20,
        land_pred_vis_a=90,
        water_pred_vis_r=20,
        water_pred_vis_a=120,

        seed=62706322,
        simulator: ABMSimulator = None,
        data_collect = True,
        show_vision = True
    ):
        """Create a new Prey-Pred model with the given parameters.

        Args:
            height: Height of the grid
            width: Width of the grid
            initial_land_prey: Number of land prey to start with
            initial_land_pred: Number of land predators to start with
            land_prey_reproduce: Probability of each land prey reproducing each step
            land_pred_reproduce: Probability of each land predator reproducing each step
            land_pred_gain_from_food: Energy a land predator gains from eating a prey
            grass: Whether to have the prey eat grass for energy
            grass_regrowth_time: How long it takes for a grass patch to regrow
                                once it is eaten
            land_prey_gain_from_food: Energy land prey gain from grass, if enabled
            seed: Random seed
            simulator: ABMSimulator instance for event scheduling
        """
        super().__init__(seed=seed)
        self.simulator = simulator
        simulator.setup(self)

        # Initialize model parameters
        self.x = x
        self.y = y
        self.grass = grass
        self.land_cells = []
        self.water_cells = []
        # Create grid using experimental cell space
        self.grid = OrthogonalMooreGrid(
            [self.x, self.y],
            torus=False,
            capacity=math.inf,
            random=self.random,
        )
        self.data_collect = data_collect
        self.seed = seed
        self.show_vision = show_vision

        self.lakes = lakes
        self.total_kills = total_kills

        if data_collect:
        # Set up data collection
            model_reporters = {
                "Land Predators": lambda m: len([a for a in m.agents_by_type[LandPredator] if isinstance(a, LandPredator)]),
                "Land Prey": lambda m: len([a for a in m.agents_by_type[LandPrey] if isinstance(a, LandPrey)]),
                "Water Predators": lambda m: len([a for a in m.agents_by_type[WaterPredator] if isinstance(a, WaterPredator)]),
                "Water Prey": lambda m: len([a for a in m.agents_by_type[WaterPrey] if isinstance(a, WaterPrey)]),
                "Total Kills": lambda m: m.total_kills,
            }
            if grass:
                model_reporters["Grass"] = lambda m: len(
                    m.agents_by_type[GrassPatch].select(lambda a: a.fully_grown)
                )

            self.datacollector = DataCollector(model_reporters)


        #Create land and water cells
        self.land_cells, self.water_cells = self.generateCells()

        # Create land prey:
        land_prey = LandPrey.create_agents(
            self,
            initial_land_prey,
            age=self.rng.uniform(0, MATURITY * 5),
            energy=self.rng.uniform(land_prey_max_eng / 2, land_prey_max_eng, initial_land_prey),
            hydration=self.rng.uniform(land_prey_max_hydration / 2, land_prey_max_hydration, initial_land_prey),
            p_reproduce=land_prey_reproduce,
            max_energy=land_prey_max_eng,
            max_hydration=land_prey_max_hydration,
            energy_from_food=land_prey_gain_from_food,
            hydration_from_water=land_prey_gain_from_water,
            cell = None,
            vision_range=prey_vis_r,
            vision_angle=prey_vis_a,
        )
        
        for p in land_prey:
            p.cell = self.random.choices(self.land_cells)[0]


        # Create water prey:
        water_prey = WaterPrey.create_agents(
            self,
            initial_water_prey,
            age=self.rng.uniform(0, MATURITY * 5),
            energy=self.rng.uniform(water_prey_max_eng / 2, water_prey_max_eng, initial_water_prey),
            p_reproduce=water_prey_reproduce,
            max_energy=water_prey_max_eng,
            max_hydration=water_prey_max_hydration,
            energy_from_food=water_prey_gain_from_food,
            hydration_from_water=water_prey_max_hydration,
            cell = None,
            vision_range=prey_vis_r,
            vision_angle=prey_vis_a,
        )
        
        for p in water_prey:
            p.cell = self.random.choices(self.water_cells)[0]

        # Create land predators:
        land_pred = LandPredator.create_agents(
            self,
            initial_land_pred,
            age=self.rng.uniform(0, MATURITY * 5),
            energy=self.rng.uniform(land_pred_max_eng / 2, land_pred_max_eng, initial_land_pred),
            hydration=self.rng.uniform(land_pred_max_hydration / 2, land_pred_max_hydration, initial_land_pred),
            p_reproduce=land_pred_reproduce,
            max_energy=land_pred_max_eng,
            max_hydration=land_pred_max_hydration,
            energy_from_food=land_pred_gain_from_food,
            hydration_from_water=land_pred_gain_from_water,
            cell=None,
            vision_range=land_pred_vis_r,
            vision_angle=land_pred_vis_a
        )

        for p in land_pred:
            p.cell = self.random.choices(self.land_cells)[0]


        # Create water predators:
        water_pred = WaterPredator.create_agents(
            self,
            initial_water_pred,
            age=self.rng.uniform(0, MATURITY * 5),
            energy=self.rng.uniform(water_pred_max_eng / 2, water_pred_max_eng, initial_water_pred),
            p_reproduce=water_pred_reproduce,
            max_energy=water_pred_max_eng,
            max_hydration=water_pred_max_hydration,
            energy_from_food=water_pred_gain_from_food,
            hydration_from_water=water_pred_max_hydration,
            cell=None,
            vision_range=water_pred_vis_r,
            vision_angle=water_pred_vis_a
        )

        for p in water_pred:
            p.cell = self.random.choices(self.water_cells)[0]

        # Create grass patches if enabled
        if grass:
            possibly_fully_grown = [True, False]
            for cell in self.grid:
                fully_grown = self.random.choice(possibly_fully_grown)
                countdown = (
                    0 if fully_grown else self.random.randrange(0, grass_regrowth_time)
                )
                GrassPatch(self, countdown, grass_regrowth_time, cell)
                    

        # Collect initial data
        self.running = True
        self.finished = False
        if data_collect:

            self.datacollector.collect(self)

            self.land_pred_hm = np.zeros((x, y))
            self.land_prey_hm = np.zeros((x, y))
            self.water_pred_hm = np.zeros((x, y))
            self.water_prey_hm = np.zeros((x, y))


    def step(self):
        """Execute one step of the model."""
        # First activate all land prey, then all predators, both in random order
        for vis in list(self.agents_by_type.get(VisionPatch, [])):
            vis.remove()

        self.agents_by_type[LandPrey].shuffle_do("step")
        self.agents_by_type[LandPredator].shuffle_do("step")
        self.agents_by_type[WaterPrey].shuffle_do("step")
        self.agents_by_type[WaterPredator].shuffle_do("step")

        # Collect data
        if self.data_collect:
            self.datacollector.collect(self)

            for agent in self.agents_by_type[LandPredator]:
                x, y = (agent.cell.coordinate[0], agent.cell.coordinate[1])
                self.land_pred_hm[y][x] += 1

            for agent in self.agents_by_type[LandPrey]:
                x, y = (agent.cell.coordinate[0], agent.cell.coordinate[1])
                self.land_prey_hm[y][x] += 1

            for agent in self.agents_by_type[WaterPredator]:
                x, y = (agent.cell.coordinate[0], agent.cell.coordinate[1])
                self.water_pred_hm[y][x] += 1

            for agent in self.agents_by_type[WaterPrey]:
                x, y = (agent.cell.coordinate[0], agent.cell.coordinate[1])
                self.water_prey_hm[y][x] += 1

            if (len(self.agents_by_type[LandPredator]) == 0 or len(self.agents_by_type[LandPrey]) == 0 or\
                len(self.agents_by_type[WaterPredator]) == 0 or len(self.agents_by_type[WaterPrey]) == 0) and not self.finished:

                    self.finished = True

                    self.land_prey_hm /= self.land_prey_hm.sum()
                    self.land_pred_hm /= self.land_pred_hm.sum()
                    self.water_prey_hm /= self.water_prey_hm.sum()
                    self.water_pred_hm /= self.water_pred_hm.sum()

                    def save_heatmap(data, title, filename):
                            fig, ax = plt.subplots()
                            ax.imshow(data, origin="lower", cmap="viridis", vmin=0, vmax=data.max())
                            ax.set_title(title)
                            ax.set_xlabel("X Position")
                            ax.set_ylabel("Y Position")
                            plt.colorbar(ax.imshow(data, origin="lower", cmap="viridis", vmin=0, vmax=data.max()), ax=ax)
                            plt.tight_layout()
                            plt.savefig(os.path.join("heatmaps", filename))
                            plt.close(fig)

                    # Export heatmaps as PNGs
                    save_heatmap(self.land_prey_hm, "Land Prey Density", "land_prey_hm.png")
                    save_heatmap(self.land_pred_hm, "Land Predator Density", "land_pred_hm.png")
                    save_heatmap(self.water_prey_hm, "Water Prey Density", "water_prey_hm.png")
                    save_heatmap(self.water_pred_hm, "Water Predator Density", "water_pred_hm.png")






    def generateCells(self):

        land_cells = []
        water_cells = []

        if self.lakes:
            noise = PerlinNoise(octaves=4, seed=int(self.seed))
            for cell in self.grid:
                x = cell.coordinate[0]
                y = cell.coordinate[1]
                if noise([x/self.x, y/self.y]) > 0.023:
                    WaterPatch(self, cell)
                    water_cells.append(cell)
                else:
                    land_cells.append(cell)
        
        else:
            land_cells = self.grid.all_cells.cells

        return land_cells, water_cells
