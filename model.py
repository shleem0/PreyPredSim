import math

from mesa import Model
from mesa.datacollection import DataCollector
from mesa.discrete_space import OrthogonalMooreGrid
from agents import GrassPatch, WaterPatch, VisionPatch, LandPrey, LandPredator
from mesa.experimental.devs import ABMSimulator

from perlin_noise import PerlinNoise


class PreyPred(Model):
    """Prey/Predator Model.

    A model for simulating predator-prey ecosystem modelling.
    """

    description = (
        "A model for simulating predator-prey ecosystem modelling."
    )

    def __init__(
        self,
        x=20,
        y=20,
        z=2,
        initial_land_prey=30,
        prey_max_eng=200,
        prey_max_hydration=200,
        initial_land_pred=5,
        land_prey_reproduce=1.0,
        pred_max_eng=200,
        pred_max_hydration=200,
        land_pred_reproduce=1.0,
        land_pred_gain_from_food=50,
        land_pred_gain_from_water=50,
        grass = True,
        grass_regrowth_time=20,
        lakes = False,
        land_prey_gain_from_food=25,
        land_prey_gain_from_water=50,
        prey_vis_r=4,
        prey_vis_a=180,
        pred_vis_r=8,
        pred_vis_a=90,
        seed=0,
        simulator: ABMSimulator = None,
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
        self.simulator.setup(self)

        # Initialize model parameters
        self.x = x
        self.y = y
        self.grass = grass
        self.land_cells = []
        # Create grid using experimental cell space
        self.grid = OrthogonalMooreGrid(
            [self.x, self.y],
            torus=False,
            capacity=math.inf,
            random=self.random,
        )

        self.lakes = lakes

        # Set up data collection
        model_reporters = {
            "Land Predators": lambda m: len(m.agents_by_type[LandPredator]),
            "Land Prey": lambda m: len(m.agents_by_type[LandPrey]),
        }
        if grass:
            model_reporters["Grass"] = lambda m: len(
                m.agents_by_type[GrassPatch].select(lambda a: a.fully_grown)
            )

        self.datacollector = DataCollector(model_reporters)

        #Create water

        self.land_cells = self.generateCells()


        # Create land prey:
        prey = LandPrey.create_agents(
            self,
            initial_land_prey,
            age=0,
            energy=self.rng.uniform(20, prey_max_eng, initial_land_prey),
            hydration=self.rng.uniform(20, prey_max_hydration, initial_land_prey),
            p_reproduce=land_prey_reproduce,
            max_energy=prey_max_eng,
            max_hydration=prey_max_hydration,
            energy_from_food=land_prey_gain_from_food,
            hydration_from_water=land_prey_gain_from_water,
            cell = None,
            vision_range=prey_vis_r,
            vision_angle=prey_vis_a,
            rep_count = 0
        )
        
        for p in prey:
            p.cell = self.random.choices(self.land_cells)[0]

        # Create land predators:
        pred = LandPredator.create_agents(
            self,
            initial_land_pred,
            age=0,
            rep_count=0,
            energy=self.rng.uniform(20, pred_max_eng, initial_land_pred),
            hydration=self.rng.uniform(20, pred_max_hydration, initial_land_pred),
            p_reproduce=land_pred_reproduce,
            max_energy=pred_max_eng,
            max_hydration=pred_max_hydration,
            energy_from_food=land_pred_gain_from_food,
            hydration_from_water=land_pred_gain_from_water,
            cell=None,
            vision_range=pred_vis_r,
            vision_angle=pred_vis_a
        )

        for p in pred:
            p.cell = self.random.choices(self.land_cells)[0]

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
        self.datacollector.collect(self)

    def step(self):
        """Execute one step of the model."""
        # First activate all land prey, then all predators, both in random order
        for vis in list(self.agents_by_type.get(VisionPatch, [])):
            vis.remove()
        self.agents_by_type[LandPrey].shuffle_do("step")
        self.agents_by_type[LandPredator].shuffle_do("step")

        # Collect data
        self.datacollector.collect(self)



    def generateCells(self):

        land_cells = []

        if self.lakes:
            noise = PerlinNoise(octaves=4, seed=int(self.seed))
            for cell in self.grid:
                x = cell.coordinate[0]
                y = cell.coordinate[1]
                if noise([x/self.x, y/self.y]) > 0.035:
                    WaterPatch(self, cell)
                else:
                    land_cells.append(cell)
        
        else:
            land_cells = self.grid.all_cells.cells

        return land_cells
