from mesa.discrete_space import CellAgent, FixedAgent
import math

from constants import DIRECTION_VECTORS, MATURITY, MAX_AGE

import torch
import torch.nn as nn
import torch.nn.functional as func
import numpy as np

import os.path

#---Helper functions---
def normalise(x, y):
    mag = np.sqrt(x * x + y * y)
    if mag < 1e-6:
        return 0.0, 0.0
    return x/mag, y/mag



#---Animal neural nets---
class LandPreyNet(nn.Module):
    def __init__(
            self, input_size=14, hidden_size=28, n_actions = 5, n_dirs = 8
    ):
        #Initialise Land Prey neural net
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, n_actions + n_dirs)

    def forward(self, x):

        x = func.tanh(self.l1(x))
        x = func.tanh(self.l2(x))

        logits = self.out(x)
        return logits


class WaterPreyNet(nn.Module):
    def __init__(
            self, input_size=10, hidden_size=20, n_actions = 4, n_dirs = 8
    ):
        #Initialise Water Prey neural net
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, n_actions + n_dirs)

    def forward(self, x):

        x = func.tanh(self.l1(x))
        x = func.tanh(self.l2(x))

        logits = self.out(x)
        return logits  


class LandPredNet(nn.Module):
    def __init__(
            self, input_size=10, hidden_size=20, n_actions = 5, n_dirs = 8
    ):
        #Initialise Land Pred neural net
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, n_actions + n_dirs)

    def forward(self, x):

        x = func.relu(self.l1(x))
        x = func.relu(self.l2(x))

        logits = self.out(x)
        return logits
    
class WaterPredNet(nn.Module):
    def __init__(
            self, input_size=5, hidden_size=10, n_actions = 4, n_dirs = 8
    ):

        #Initialise Water Pred neural net
        super().__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, n_actions + n_dirs)

    def forward(self, x):

        x = func.relu(self.l1(x))
        x = func.relu(self.l2(x))

        logits = self.out(x)
        return logits
    
#---Animal agents---
class Animal(CellAgent):
    """The base animal class."""

    def __init__(
        self, model, age, energy, hydration, p_reproduce, max_energy, max_hydration, energy_from_food, hydration_from_water, cell=None, 
        vision_range=0, vision_angle=0, vision=[], heading=(0,1), nn=None
    ):
        """Initialize an animal.

        Args:
            model: Model instance
            energy: Starting amount of energy
            p_reproduce: Probability of reproduction (asexual)
            energy_from_food: Energy obtained from 1 unit of food
            cell: Cell in which the animal starts
        """

        super().__init__(model)
        self.age = 0
        self.max_age = MAX_AGE
        self.rep_count = 0
        self.drank = 0
        self.energy = energy
        self.hydration = hydration
        self.p_reproduce = p_reproduce
        self.max_energy = max_energy
        self.max_hydration = max_hydration
        self.energy_from_food = energy_from_food
        self.hydration_from_water = hydration_from_water
        self.cell = cell
        self.vision_range = vision_range
        self.vision_angle = vision_angle
        self.vision = vision
        self.heading = (0, 1) #default: north
        self.prev_opponent_vec = (0, 0),
        self.nn = nn
        self.input_vector = []


    def closest_water(self):
        visible = self.vision

        cells_with_water = [cell for cell in visible
            if any(isinstance(obj, WaterPatch) for obj in cell.agents)]

        if cells_with_water:
            closest_cell = min(cells_with_water, key=lambda c: self.cell_distance(self.cell, c))
            min_dist = self.cell_distance(self.cell, closest_cell)
            count = len(cells_with_water)

            return closest_cell.coordinate[0], closest_cell.coordinate[1], min_dist, count
        else:
            return (0, 0, self.vision_range, 0)
        

    def get_genome(self):
        return torch.nn.utils.parameters_to_vector(self.nn.parameters()).detach().cpu().numpy().copy()

    def set_genome(self, genome):
        vec = torch.tensor(genome, dtype=torch.float32)
        nn.utils.vector_to_parameters(vec, self.nn.parameters())


    def move_to(self, cell):
    #Moves agent and updates heading appropriately

        if self.cell is not None:
            dx = cell.coordinate[0] - self.cell.coordinate[0]
            dy = cell.coordinate[1] - self.cell.coordinate[1]

            if (dx, dy) != (0, 0):
                self.heading = (dx, dy)

        self.cell = cell
        self.vision = self.visible_cells()

    def get_cell_in_direction(self, dir_idx):
        """Return the neighbouring cell in the absolute direction index (or None)."""
        if dir_idx < 0 or dir_idx >= len(DIRECTION_VECTORS):
            return None

        dx, dy = DIRECTION_VECTORS[dir_idx]
        cx, cy = self.cell.coordinate
        target_coord = (cx + dx, cy + dy)

        for c in self.cell.neighborhood:
            if c.coordinate == target_coord:
                return c
        return None

    def _valid_move_mask(self):
        """Return a boolean mask (list) of length len(DIRECTION_VECTORS) where True means that
        the direction exists and is not water-blocked."""
        mask = []
        for (dx, dy) in DIRECTION_VECTORS:
            cx, cy = self.cell.coordinate
            target_coord = (cx + dx, cy + dy)
            found = None
            for cell in self.cell.neighborhood:
                if cell.coordinate == target_coord:
                    found = cell
                    break
            if found is None:
                mask.append(False)
            elif (isinstance(self, LandPrey) or isinstance(self, LandPredator)) and any(isinstance(o, WaterPatch) for o in found.agents):
                mask.append(False)
            elif (isinstance(self, WaterPrey) or isinstance(self, WaterPredator)) and not (any(isinstance(o, WaterPatch) for o in found.agents)):
                mask.append(False)
            else:
                mask.append(True)
        return mask


    def visible_cells(self):
        neighbours = self.cell.get_neighborhood(radius=self.vision_range)

        if self.vision_angle >= 360 or self.heading == (0, 0):
            return list(neighbours)

        hx, hy = self.heading
        half = math.radians(self.vision_angle/2)
        cos_limit = math.cos(half)

        h_len = math.hypot(hx, hy)
        visible = []

        cx = self.cell.coordinate[0]
        cy = self.cell.coordinate[1]

        for cell in neighbours:
            dx = cell.coordinate[0] - cx
            dy = cell.coordinate[1] - cy
            dist = math.hypot(dx, dy)

            if dist == 0 or dist > self.vision_range:
                continue

            cos_theta = (dx * hx + dy * hy) / (dist * h_len)
            if cos_theta >= cos_limit:
                visible.append(cell)
            
        visible.append(self.cell)
        return visible
    

    def cell_distance(self, c1, c2):
    #get distance between cells
        x1, y1 = c1.coordinate
        x2, y2 = c2.coordinate
        return math.hypot(x1 - x2, y1 - y2)
    
    def spawn_offspring(self):
        """Create offspring by splitting energy and creating new instance"""


    def feed(self):
        """Abstract method to be implemented by subclasses."""

    def drink(self):
        
        vision = self.visible_cells()

        neighbouring_water = list(self.cell.neighborhood.select(
            lambda cell: any ((isinstance(obj, WaterPatch) for obj in cell.agents))
        ))

        neighbouring_visible_water = [cell for cell in vision if cell in neighbouring_water]

        if neighbouring_visible_water:
            self.drank += 1
            self.hydration += self.hydration_from_water

    def move_in_direction(self, dirs):
        """Abstract method to be implemented by subclasses"""

    def update_input_vector(self):
        """Abstract method to be implemented by subclasses"""


    def step(self):
        """Execute one step of the animal's behavior."""

        self.energy -= 2
        if self.model.lakes == True and (isinstance(self, LandPredator) or isinstance(self, LandPrey)):
            self.hydration -= 2
        self.age += 1

        if self.energy < 0 or self.hydration < 0 or self.age > self.max_age:
            self.remove()
            #print(f"{self.__class__} died: energy={self.energy}, hydration={self.hydration}, age= {self.age}")
            return
        
        self.vision = self.visible_cells()
        self.update_input_vector()

        x = torch.tensor(self.input_vector, dtype=torch.float32).unsqueeze(0)
        logits = self.nn(x)[0]

        if isinstance(self, LandPredator) or isinstance(self, LandPrey):
            act_logits = logits[:5]
            dir_logits = logits[5:]
        else:
            act_logits = logits[:4]
            dir_logits = logits[4:]


        if isinstance(self, Predator):
            prey_in_cell = []
            prey_in_cell.extend([a for a in self.cell.agents if isinstance(a, Prey)])

            if prey_in_cell:  # If there are any prey present
                prey_to_eat = self.random.choice(prey_in_cell)

                self.energy = min(self.energy + self.energy_from_food, self.max_energy)
                self.prey_eaten += (1 - self.energy / self.max_energy) 
                self.model.total_kills += 1

                #print(f"{prey_to_eat.__class__} eaten by {self.__class__}")
                prey_to_eat.remove()

                return



        act_probs = func.softmax(act_logits, dim=0)
        action = torch.multinomial(act_probs, 1).item()

        if action == 0:
            self.move_in_direction(dir_logits)

            heading_norm = normalise(self.heading[0], self.heading[1])
            alignment = heading_norm[0] * self.prev_opponent_vec[0] + heading_norm[1] * self.prev_opponent_vec[1]

            if isinstance(self, Predator):
                self.chase += max(0, alignment)
                self.chase -= 0.3 * max(0, -alignment)
            else:
                self.flee += max(0, -alignment)
                self.flee -= 0.3 * max(0, alignment)


        elif action == 1:
            self.feed()

        elif action == 3:
            self.spawn_offspring()

        elif action == 4:
            pass

        elif action == 5:
            self.drink()

        if self.model.show_vision:
            for cell in self.vision:
                if isinstance(self, Prey):
                    VisionPatch(self.model, cell, Prey)
                else:
                    VisionPatch(self.model, cell, Predator)



#---Prey---

class Prey(Animal):

    def __init__(
    self,
    model,
    age,
    energy,
    hydration,
    p_reproduce,
    max_energy,
    max_hydration,
    energy_from_food,
    hydration_from_water,
    cell,
    vision_range=3,
    vision_angle=180,
    vision=[],
    heading=(0,1),
    prev_opponent_vec = (0, 0),
    nn=LandPreyNet()
    ):
        super().__init__(
            model=model,
            energy=energy,
            age=age,
            hydration=hydration,
            p_reproduce=p_reproduce,
            max_energy=max_energy,
            max_hydration=max_hydration,
            energy_from_food=energy_from_food,
            hydration_from_water=hydration_from_water,
            cell=cell,
            vision_range=vision_range,
            vision_angle=vision_angle,
            nn=nn
            )



    #---Input vector helpers---
    def closest_food(self):
        visible = self.vision + [self.cell]

        cells_with_grass = [cell for cell in visible
            if any(isinstance(obj, GrassPatch) and obj.fully_grown for obj in cell.agents)]

        if cells_with_grass:
            closest_cell = min(cells_with_grass, key=lambda c: self.cell_distance(self.cell, c))
            min_dist = self.cell_distance(self.cell, closest_cell)
            count = len(cells_with_grass)

            return closest_cell.coordinate[0], closest_cell.coordinate[1], min_dist, count

        else:
            return (0, 0, self.vision_range, 0)
        
    def closest_pred(self):
        visible = self.vision

        cells_with_pred = [cell for cell in visible
            if any(isinstance(obj, Predator) for obj in cell.agents)]

        if cells_with_pred:
            closest_cell = min(cells_with_pred, key=lambda c: self.cell_distance(self.cell, c))
            min_dist = self.cell_distance(self.cell, closest_cell)
            count = len(cells_with_pred)

            return closest_cell.coordinate[0], closest_cell.coordinate[1], min_dist, count
        else:
            return (0, 0, self.vision_range, 0)


class LandPrey(Prey):
    """A land prey animal that walks around, reproduces (asexually) and gets eaten."""

    def __init__(
        self,
        model,
        age,
        energy,
        hydration,
        p_reproduce,
        max_energy,
        max_hydration,
        energy_from_food,
        hydration_from_water,
        cell,
        vision_range=3,
        vision_angle=180,
        vision=[],
        heading=(0,1),
        prev_opponent_vec=(0,0),
        nn=LandPreyNet()
    ):
        super().__init__(
            model=model,
            energy=energy,
            age=age,
            hydration=hydration,
            p_reproduce=p_reproduce,
            max_energy=max_energy,
            max_hydration=max_hydration,
            energy_from_food=energy_from_food,
            hydration_from_water=hydration_from_water,
            cell=cell,
            vision_range=vision_range,
            vision_angle=vision_angle,
            nn=nn
        )

        self.flee = 0
        self.food_approach = 0
        self.food_eaten = 0
        self.water_approach = 0
        if os.path.isfile("prey_net.pth"):
            self.nn.load_state_dict(torch.load("prey_net.pth"))
        else:
            nn = LandPreyNet()


    def update_input_vector(self):

        if self.input_vector:
            prev_pred_dist = self.input_vector[10]
            prev_food_dist = self.input_vector[6]
            prev_water_dist = self.input_vector[2]
        else:
            prev_pred_dist = 0
            prev_food_dist = 0
            prev_water_dist = 0

        p_dx, p_dy, min_pred_dist, n_pred = self.closest_pred()
        f_dx, f_dy, min_food_dist, n_food = self.closest_food()
        w_dx, w_dy, min_water_dist, n_water = self.closest_water()

        w_dx, w_dy = normalise(w_dx, w_dy)
        p_dx, p_dy = normalise(p_dx, p_dy)
        self.prev_opponent_vec = (p_dx, p_dy)
        f_dx, f_dy = normalise(f_dx, f_dy)

        n_visible_cells = max(1, len(self.visible_cells()))

        if prev_food_dist < (min_food_dist / self.vision_range):
            self.food_approach += 1

        if prev_water_dist < (min_water_dist / self.vision_range):
            self.water_approach += 1

        self.input_vector = [
            self.energy / self.max_energy,
            self.hydration / self.max_hydration,

            min_water_dist / self.vision_range,
            n_water / n_visible_cells,
            w_dx,
            w_dy,

            min_food_dist / self.vision_range,
            n_food / n_visible_cells,
            f_dx,
            f_dy,

            min_pred_dist / self.vision_range,
            n_pred / n_visible_cells,
            p_dx,
            p_dy
        ]

    def spawn_offspring(self):
        #Create offspring by splitting energy and creating new instance.

        if self.random.random() < self.p_reproduce and self.energy >= self.max_energy * 0.5 and self.age >= MATURITY:

            self.rep_count += 1

            split_ratio = 0.3 
            child_energy = max(self.energy * split_ratio, self.max_energy * 0.1)
            self.energy = max(self.energy * (1 - split_ratio), self.max_energy * 0.1)

            child_nn = LandPreyNet()
            if os.path.isfile("land_prey_net.pth"):
                child_nn.load_state_dict(torch.load("land_prey_net.pth"))

            genome = self.get_genome()

            mut_rate = 0.0
            mut_str = 0.0
            mask = np.random.rand(genome.size) < mut_rate
            genome[mask] += np.random.normal(0, mut_str, size=mask.sum())

            vec = torch.tensor(genome, dtype=torch.float32)
            nn.utils.vector_to_parameters(vec, child_nn.parameters())

            self.__class__(
                self.model,
                0,
                child_energy,
                self.model.rng.uniform(self.max_hydration / 2, self.max_hydration),
                self.p_reproduce,
                self.max_energy,
                self.max_hydration,
                self.energy_from_food,
                self.hydration_from_water,
                self.cell,
                self.vision_range,
                self.vision_angle,
                self.vision,
                self.heading,
                self.prev_opponent_vec,
                child_nn
            )


    def feed(self):
        #If possible, eat grass at current location
        grass_patch = next(
            obj for obj in self.cell.agents if isinstance(obj, GrassPatch)
        )
        if not grass_patch:
            return
        if grass_patch.fully_grown:
            self.energy = min(self.energy + self.energy_from_food, self.max_energy)
            self.food_eaten += 1
            grass_patch.fully_grown = False


    def move_in_direction(self, dirs):

        if not isinstance(dirs, torch.Tensor):
            dirs = torch.tensor(dirs, dtype=torch.float32)

        # Determine valid directions (absolute mapping)
        valid_mask = self._valid_move_mask()
        # convert mask to tensor
        mask_tensor = torch.tensor([1.0 if v else 0.0 for v in valid_mask], dtype=torch.float32)

        # If no directions are valid, fallback to any non-water neighbor (random)
        if mask_tensor.sum().item() == 0:
            candidates = [cell for cell in self.cell.neighborhood
                          if not any(isinstance(o, WaterPatch) for o in cell.agents)]
            if not candidates:
                return
            chosen = self.random.choice(candidates)
            self.move_to(chosen)
            return

        # Mask invalid logits by assigning a large negative value so softmax ~ 0
        LARGE_NEG = -1e9
        dirs_masked = dirs.clone()
        for i, valid in enumerate(valid_mask):
            if not valid:
                # keep shape-safe
                dirs_masked[i] = LARGE_NEG

        probs = func.softmax(dirs_masked, dim=0)
        # Sample one direction index
        try:
            idx = torch.multinomial(probs, 1).item()
        except Exception:
            # numerical fallback: argmax
            idx = int(torch.argmax(probs).item())

        target_cell = self.get_cell_in_direction(idx)
        if target_cell is None:
            # If something went wrong, fallback to random valid neighbor
            candidates = [cell for cell in self.cell.neighborhood
                          if not any(isinstance(o, WaterPatch) for o in cell.agents)]
            if not candidates:
                return
            chosen = self.random.choice(candidates)
            self.move_to(chosen)
            return
        
        self.move_to(target_cell)



class WaterPrey(Prey):

    def __init__(
        self,
        model,
        age,
        energy,
        p_reproduce,
        max_energy,
        max_hydration,
        energy_from_food,
        hydration_from_water,
        cell,
        vision_range=3,
        vision_angle=180,
        vision=[],
        heading=(0,1),
        prev_opponent_vec=(0,0),
        nn=WaterPreyNet()
    ):
        super().__init__(
            model=model,
            energy=energy,
            age=age,
            hydration=max_hydration,
            p_reproduce=p_reproduce,
            max_energy=max_energy,
            max_hydration=max_hydration,
            energy_from_food=energy_from_food,
            hydration_from_water=hydration_from_water,
            cell=cell,
            vision_range=vision_range,
            vision_angle=vision_angle,
            nn=nn
        )

        self.flee = 0
        self.food_approach = 0
        self.food_eaten = 0
        if os.path.isfile("water_prey_net.pth"):
            self.nn.load_state_dict(torch.load("water_prey_net.pth"))
        else:
            nn = WaterPreyNet()


    def update_input_vector(self):

        if self.input_vector:
            prev_pred_dist = self.input_vector[6]
            prev_food_dist = self.input_vector[2]
        else:
            prev_pred_dist = 0
            prev_food_dist = 0

        p_dx, p_dy, min_pred_dist, n_pred = self.closest_pred()
        f_dx, f_dy, min_food_dist, n_food = self.closest_food()

        p_dx, p_dy = normalise(p_dx, p_dy)
        self.prev_opponent_vec = (p_dx, p_dy)
        f_dx, f_dy = normalise(f_dx, f_dy)

        n_visible_cells = max(1, len(self.visible_cells()))

        if prev_food_dist < (min_food_dist / self.vision_range):
            self.food_approach += 1

        self.input_vector = [
            self.energy / self.max_energy,
            self.hydration / self.max_hydration,

            min_food_dist / self.vision_range,
            n_food / n_visible_cells,
            f_dx,
            f_dy,

            min_pred_dist / self.vision_range,
            n_pred / n_visible_cells,
            p_dx,
            p_dy
        ]


    def feed(self):
        #If possible, eat grass at current location
        grass_patch = next(
            obj for obj in self.cell.agents if isinstance(obj, GrassPatch)
        )
        if not grass_patch:
            return
        if grass_patch.fully_grown:
            self.energy = min(self.energy + self.energy_from_food, self.max_energy)
            self.food_eaten += 1
            grass_patch.fully_grown = False


    def spawn_offspring(self):
        #Create offspring by splitting energy and creating new instance.

        if self.random.random() < self.p_reproduce and self.energy >= self.max_energy * 0.5 and self.age >= MATURITY:

            self.rep_count += 1

            split_ratio = 0.3 
            child_energy = max(self.energy * split_ratio, self.max_energy * 0.1)
            self.energy = max(self.energy * (1 - split_ratio), self.max_energy * 0.1)

            child_nn = WaterPreyNet()
            if os.path.isfile("water_prey_net.pth"):
                child_nn.load_state_dict(torch.load("water_prey_net.pth"))

            genome = self.get_genome()

            mut_rate = 0.0
            mut_str = 0.0
            mask = np.random.rand(genome.size) < mut_rate
            genome[mask] += np.random.normal(0, mut_str, size=mask.sum())

            vec = torch.tensor(genome, dtype=torch.float32)
            nn.utils.vector_to_parameters(vec, child_nn.parameters())

            self.__class__(
                self.model,
                0,
                child_energy,
                self.p_reproduce,
                self.max_energy,
                self.max_hydration,
                self.energy_from_food,
                self.hydration_from_water,
                self.cell,
                self.vision_range,
                self.vision_angle,
                self.vision,
                self.heading,
                self.prev_opponent_vec,
                child_nn
            )

    def move_in_direction(self, dirs):

        if not isinstance(dirs, torch.Tensor):
            dirs = torch.tensor(dirs, dtype=torch.float32)

        # Determine valid directions (absolute mapping)
        valid_mask = self._valid_move_mask()
        # convert mask to tensor
        mask_tensor = torch.tensor([1.0 if v else 0.0 for v in valid_mask], dtype=torch.float32)

        # If no directions are valid, fallback to any non-water neighbor (random)
        if mask_tensor.sum().item() == 0:
            candidates = [cell for cell in self.cell.neighborhood
                          if any(isinstance(o, WaterPatch) for o in cell.agents)]
            if not candidates:
                return
            chosen = self.random.choice(candidates)
            self.move_to(chosen)
            return

        # Mask invalid logits by assigning a large negative value so softmax ~ 0
        LARGE_NEG = -1e9
        dirs_masked = dirs.clone()
        for i, valid in enumerate(valid_mask):
            if not valid:
                # keep shape-safe
                dirs_masked[i] = LARGE_NEG

        probs = func.softmax(dirs_masked, dim=0)
        # Sample one direction index
        try:
            idx = torch.multinomial(probs, 1).item()
        except Exception:
            # numerical fallback: argmax
            idx = int(torch.argmax(probs).item())

        target_cell = self.get_cell_in_direction(idx)
        if target_cell is None:
            # If something went wrong, fallback to random valid neighbor
            candidates = [cell for cell in self.cell.neighborhood
                          if any(isinstance(o, WaterPatch) for o in cell.agents)]
            if not candidates:
                return
            chosen = self.random.choice(candidates)
            self.move_to(chosen)
            return
        
        self.move_to(target_cell)


#---Predators---
class Predator(Animal):

    def __init__(
        self,
        model,
        age,
        energy,
        hydration,
        p_reproduce,
        max_energy,
        max_hydration,
        energy_from_food,
        hydration_from_water,
        cell,
        nn,
        vision_range=6,
        vision_angle=90,
        vision=[],
        heading=(0,1),
        prev_opponent_vec=(0,0)
    ):
        super().__init__(
            model=model,
            energy=energy,
            age=age,
            hydration=hydration,
            p_reproduce=p_reproduce,
            max_energy=max_energy,
            max_hydration=max_hydration,
            energy_from_food=energy_from_food,
            hydration_from_water=hydration_from_water,
            cell=cell,
            vision_range=vision_range,
            vision_angle=vision_angle,
            nn = nn
        )

        self.prey_eaten = 0
        self.chase = 0

    def closest_prey(self):
        visible = self.vision

        cells_with_prey = [cell for cell in visible
            if any(isinstance(obj, Prey) for obj in cell.agents)]

        if cells_with_prey:
            closest_cell = min(cells_with_prey, key=lambda c: self.cell_distance(self.cell, c))
            min_dist = self.cell_distance(self.cell, closest_cell)
            count = len(cells_with_prey)

            return closest_cell.coordinate[0], closest_cell.coordinate[1], min_dist, count
        else:
            return (0, 0, self.vision_range, 0)


class LandPredator(Predator):
    """A land predator that walks around, reproduces (asexually) and eats prey."""

    def __init__(
        self,
        model,
        age,
        energy,
        hydration,
        p_reproduce,
        max_energy,
        max_hydration,
        energy_from_food,
        hydration_from_water,
        cell,
        vision_range=6,
        vision_angle=90,
        vision=[],
        heading=(0,1),
        prev_opponent_vec=(0,0),
        nn=LandPredNet()
    ):
        super().__init__(
            model=model,
            energy=energy,
            age=age,
            hydration=hydration,
            p_reproduce=p_reproduce,
            max_energy=max_energy,
            max_hydration=max_hydration,
            energy_from_food=energy_from_food,
            hydration_from_water=hydration_from_water,
            cell=cell,
            vision_range=vision_range,
            vision_angle=vision_angle,
            nn = nn
        )

        self.prey_eaten = 0
        self.chase = 0
        if os.path.isfile("land_pred_net.pth"):
            self.nn.load_state_dict(torch.load("land_pred_net.pth"))
        else:
            self.nn = LandPredNet()
            


    def update_input_vector(self):

        if self.input_vector:
            prev_prey_dist = self.input_vector[6]
        else:
            prev_prey_dist = self.vision_range

        w_dx, w_dy, min_water_dist, n_water = self.closest_water()
        p_dx, p_dy, min_prey_dist, n_prey = self.closest_prey()
        n_visible_cells = max(1, len(self.visible_cells()))

        w_dx, w_dy = normalise(w_dx, w_dy)
        p_dx, p_dy = normalise(p_dx, p_dy)
        self.prev_opponent_vec = (p_dx, p_dy)

        self.input_vector = [
            self.energy / self.max_energy,
            self.hydration / self.max_hydration,

            min_water_dist / self.vision_range,
            n_water / n_visible_cells,
            w_dx,
            w_dy,

            min_prey_dist / self.vision_range,
            n_prey / n_visible_cells,
            p_dx,
            p_dy,
        ]


    def spawn_offspring(self):

        if self.random.random() < self.p_reproduce and self.energy >= self.max_energy * 0.7 and self.age > MATURITY:

            self.rep_count += 1

            split_ratio = 0.3 
            child_energy = max(self.energy * split_ratio, self.max_energy * 0.3)
            self.energy = max(self.energy * (1 - split_ratio), self.max_energy * 0.3)

            child_nn = LandPredNet()
            if os.path.isfile("land_pred_net.pth"):
                child_nn.load_state_dict(torch.load("land_pred_net.pth"))

            genome = self.get_genome()

            mut_rate = 0.02
            mut_str = 0.1
            mask = np.random.rand(genome.size) < mut_rate
            genome[mask] += np.random.normal(0, mut_str, size=mask.sum())

            vec = torch.tensor(genome, dtype=torch.float32)
            nn.utils.vector_to_parameters(vec, child_nn.parameters())

            self.__class__(
                self.model,
                0,
                child_energy,
                self.model.rng.uniform(self.max_hydration / 2, self.max_hydration),
                self.p_reproduce,
                self.max_energy,
                self.max_hydration,
                self.energy_from_food,
                self.hydration_from_water,
                self.cell,
                self.vision_range,
                self.vision_angle,
                self.vision,
                self.heading,
                self.prev_opponent_vec,
                child_nn,
            )


    def feed(self):
        """If possible, eat prey at current location."""

        nearby_cells = [self.cell] + list(self.cell.neighborhood)
        
        prey = []
        for cell in nearby_cells:
            prey.extend([a for a in cell.agents if isinstance(a, Prey)])

        if prey:  # If there are any prey present
            prey_to_eat = self.random.choice(prey)
            self.energy = min(self.energy + self.energy_from_food, self.max_energy)
            self.prey_eaten += (1 - self.energy / self.max_energy) 
            self.model.total_kills += 1
            #print(f"{prey_to_eat.__class__} eaten by {self.__class__}")
            prey_to_eat.remove()

    
    def move_in_direction(self, dirs):

        candidates = [cell for cell in self.cell.neighborhood
                    if not any(isinstance(o, WaterPatch) for o in cell.agents)]
        if not candidates:
            return

        probs = func.softmax(dirs[:len(candidates)], dim=0)
        idx = torch.multinomial(probs, 1).item()
        self.move_to(candidates[idx])



class WaterPredator(Predator):

    def __init__(
        self,
        model,
        age,
        energy,
        p_reproduce,
        max_energy,
        max_hydration,
        energy_from_food,
        hydration_from_water,
        cell,
        vision_range=6,
        vision_angle=90,
        vision=[],
        heading=(0,1),
        prev_opponent_vec=(0,0),
        nn=WaterPredNet()
    ):
        super().__init__(
            model=model,
            energy=energy,
            age=age,
            hydration=max_hydration,
            p_reproduce=p_reproduce,
            max_energy=max_energy,
            max_hydration=max_hydration,
            energy_from_food=energy_from_food,
            hydration_from_water=hydration_from_water,
            cell=cell,
            vision_range=vision_range,
            vision_angle=vision_angle,
            nn = nn
        )

        self.prey_eaten = 0
        self.chase = 0

        if os.path.isfile("water_pred_net.pth"):
            self.nn.load_state_dict(torch.load("water_pred_net.pth"))
        else:
            self.nn = WaterPredNet()


    def update_input_vector(self):

        if self.input_vector:
            prev_prey_dist = self.input_vector[2]
        else:
            prev_prey_dist = self.vision_range

        p_dx, p_dy, min_prey_dist, n_prey = self.closest_prey()
        n_visible_cells = max(1, len(self.visible_cells()))

        p_dx, p_dy = normalise(p_dx, p_dy)
        self.prev_opponent_vec = (p_dx, p_dy)

        self.input_vector = [
            self.energy / self.max_energy,

            min_prey_dist / self.vision_range,
            n_prey / n_visible_cells,
            p_dx,
            p_dy,
        ]


    def feed(self):
        """If possible, eat prey at current location."""

        nearby_cells = [self.cell] + list(self.cell.neighborhood)
        
        prey = []
        for cell in nearby_cells:
            prey.extend([a for a in cell.agents if isinstance(a, Prey)])

        if prey:  # If there are any prey present
            prey_to_eat = self.random.choice(prey)
            self.energy = min(self.energy + self.energy_from_food, self.max_energy)
            self.prey_eaten += (1 - self.energy / self.max_energy) 
            self.model.total_kills += 1
            prey_to_eat.remove()


    def spawn_offspring(self):

        if self.random.random() < self.p_reproduce and self.energy >= self.max_energy * 0.7 and self.age > MATURITY:

            self.rep_count += 1

            split_ratio = 0.3 
            child_energy = max(self.energy * split_ratio, self.max_energy * 0.3)
            self.energy = max(self.energy * (1 - split_ratio), self.max_energy * 0.3)

            child_nn = WaterPredNet()
            if os.path.isfile("water_pred_net.pth"):
                child_nn.load_state_dict(torch.load("water_pred_net.pth"))

            genome = self.get_genome()

            mut_rate = 0.02
            mut_str = 0.1
            mask = np.random.rand(genome.size) < mut_rate
            genome[mask] += np.random.normal(0, mut_str, size=mask.sum())

            vec = torch.tensor(genome, dtype=torch.float32)
            nn.utils.vector_to_parameters(vec, child_nn.parameters())

            self.__class__(
                self.model,
                0,
                child_energy,
                self.p_reproduce,
                self.max_energy,
                self.max_hydration,
                self.energy_from_food,
                self.hydration_from_water,
                self.cell,
                self.vision_range,
                self.vision_angle,
                self.vision,
                self.heading,
                self.prev_opponent_vec,
                child_nn,
            )


    def move_in_direction(self, dirs):

        candidates = [cell for cell in self.cell.neighborhood
                    if any(isinstance(o, WaterPatch) for o in cell.agents)]
        if not candidates:
            return

        probs = func.softmax(dirs[:len(candidates)], dim=0)
        idx = torch.multinomial(probs, 1).item()
        self.move_to(candidates[idx])


#---Map/terrain agents---

class GrassPatch(FixedAgent):
    """A patch of grass that grows at a fixed rate and can be eaten by prey."""

    @property
    def fully_grown(self):
        """Whether the grass patch is fully grown."""
        return self._fully_grown

    @fully_grown.setter
    def fully_grown(self, value: bool) -> None:
        """Set grass growth state and schedule regrowth if eaten."""
        self._fully_grown = value

        if not value:  # If grass was just eaten
            self.model.simulator.schedule_event_relative(
                setattr,
                self.grass_regrowth_time,
                function_args=[self, "fully_grown", True],
            )

    def __init__(self, model, countdown, grass_regrowth_time, cell):
        """Create a new patch of grass.

        Args:
            model: Model instance
            countdown: Time until grass is fully grown again
            grass_regrowth_time: Time needed to regrow after being eaten
            cell: Cell to which this grass patch belongs
        """
        super().__init__(model)
        self._fully_grown = countdown == 0
        self.grass_regrowth_time = grass_regrowth_time
        self.cell = cell

        # Schedule initial growth if not fully grown
        if not self.fully_grown:
            self.model.simulator.schedule_event_relative(
                setattr, countdown, function_args=[self, "fully_grown", True]
            )



class WaterPatch(FixedAgent):
    """An area of water where fish and sharks can move within"""
    def __init__(self, model, cell):
        super().__init__(model)
        self.cell = cell



class VisionPatch(CellAgent):
    def __init__(self, model, cell, creature):
        super().__init__(model)
        self.cell = cell
        self.creature = creature