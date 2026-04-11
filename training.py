import numpy as np
import torch
import os
import random
from agents import LandPrey, LandPredator, WaterPrey, WaterPredator, LandPreyNet, LandPredNet, WaterPreyNet, WaterPredNet
from model import PreyPred
from mesa.experimental.devs import ABMSimulator
import gc

class Individual:
    def __init__(self, genome):
        self.genome = genome
        self.fitness = None
        self.agent = None

    def clone(self):
        return Individual(self.genome.copy())


class GeneticAlgorithm:
    def __init__(self, nn_class, ind_num=10, pop_size=20, g1_size=5, g2_size=5, g3_size=5, mut_rate=0.075, mut_strength=0.4):
        self.nn_class = nn_class
        self.template_nn = nn_class()
        self.ind_num = ind_num
        self.pop_size = pop_size
        self.g1_size = g1_size
        self.g2_size = g2_size
        self.g3_size = g3_size
        self.mut_rate = mut_rate
        self.mut_strength = mut_strength

        base_genome = torch.nn.utils.parameters_to_vector(self.template_nn.parameters()).detach().cpu().numpy()

        # Initialise neural nets with random weights
        self.inds = [Individual(base_genome.copy()) for p in range(ind_num)]


    def evaluate_fitness(self, model_class, training_class, g1_class, g2_class, g3_class, training_args, g1_args, g2_args, g3_args, steps=500, runs_per=2):
        """Evaluate each genome by running a Mesa simulation."""
        fitnesses = []

        for ind in self.inds:

            genome_fitness = 0

            for run in range(runs_per):
                model = model_class(simulator=ABMSimulator(), initial_land_prey=0, initial_land_pred=0, initial_water_prey = 0, initial_water_pred = 0, data_collect = False, seed=random.randint(1, 9999))  # Reset model

                training_agents = []
                for i in range(self.pop_size):
                    if training_class == LandPredator or training_class == LandPrey:
                        start_terrain = random.choice(model.land_cells)
                    else:
                        start_terrain = random.choice(model.water_cells)

                    nn = self.nn_class()
                    genome_tensor = torch.from_numpy(ind.genome).float()
                    torch.nn.utils.vector_to_parameters(
                        genome_tensor,
                        nn.parameters()
                    )
                    training_agents.append(training_class(nn=nn, model = model, cell=start_terrain, **training_args))

                for i in range(self.g1_size):
                    if g1_class == LandPredator or g1_class == LandPrey:
                        start_terrain = random.choice(model.land_cells)
                    else:
                        start_terrain = random.choice(model.water_cells)

                    g1_agent = g1_class(model=model, cell=start_terrain, **g1_args)

                for i in range(self.g2_size):
                    if g2_class == LandPredator or g2_class == LandPrey:
                        start_terrain = random.choice(model.land_cells)
                    else:
                        start_terrain = random.choice(model.water_cells)

                    g2_agent = g2_class(model=model, cell=start_terrain, **g2_args)

                for i in range(self.g3_size):
                    if g3_class == LandPredator or g3_class == LandPrey:
                        start_terrain = random.choice(model.land_cells)
                    else:
                        start_terrain = random.choice(model.water_cells)

                    g3_agent = g3_class(model=model, cell=start_terrain, **g3_args)

                for s in range(steps):
                    if len(model.agents_by_type[training_class]) == 0:
                        break
                    model.step()

                run_fitness = 0
                for training_agent in training_agents:
                    if training_agent.energy <= 1:
                        death_penalty = 0.33
                    else:
                        death_penalty = 1

                    #Prey fitness - survival + eating/drinking + reproduction + foraging + fleeing - death
                    if training_class == LandPrey:
                        foraging = training_agent.food_approach + training_agent.water_approach

                        fitness = (training_agent.age +
                                    1.5 * training_agent.food_eaten +
                                    0.5 * training_agent.drank +
                                    2 * training_agent.rep_count +
                                    foraging +
                                    training_agent.flee
                                    ) * death_penalty

                    elif training_class == WaterPrey:
                        fitness = (training_agent.age +
                                    1.5 * training_agent.food_eaten +
                                    2 * training_agent.rep_count+
                                    training_agent.food_approach +
                                    training_agent.flee
                                    ) * death_penalty
                                        
                    #Predator fitness - survival + eating/drinking + reproduction + moving + chasing  - death
                    elif training_class == LandPredator:
                        fitness = (2 * training_agent.age +
                                    3 * (training_agent.prey_eaten) +
                                    1.5 * training_agent.drank +
                                    1.5 * (training_agent.rep_count * training_agent.energy / training_agent.max_energy)+
                                    0.75 * training_agent.chase
                                    ) * death_penalty
                        
                    elif training_class == WaterPredator:
                        fitness = (2 * training_agent.age +
                                    3 * (training_agent.prey_eaten) +
                                    1.5 * (training_agent.rep_count * training_agent.energy / training_agent.max_energy)+
                                    0.75 * training_agent.chase
                                    ) * death_penalty
                        
                    run_fitness += fitness

                genome_fitness += (run_fitness / self.pop_size)
                
            ind.fitness = genome_fitness / runs_per
            fitnesses.append(ind.fitness)

        return np.array(fitnesses)



    def select_parents(self, fitnesses):
        
        min_fit = np.min(fitnesses)
        if min_fit < 0:
            fitnesses = fitnesses - min_fit + 1e-6  # shift up slightly

        probs = fitnesses / np.sum(fitnesses)
        idxs = np.random.choice(len(self.inds), size=2, p=probs)
        return self.inds[idxs[0]], self.inds[idxs[1]]

    def crossover(self, genome1, genome2):
        point = np.random.randint(0, len(genome1))
        child1 = np.concatenate([genome1[:point], genome2[point:]])
        child2 = np.concatenate([genome2[:point], genome1[point:]])
        return child1, child2

    def mutate(self, genome):
        n = int(self.mut_rate * len(genome))
        if n == 0:
            return genome
        idx = np.random.choice(len(genome), n, replace=False)
        genome[idx] += np.random.normal(0, self.mut_strength, size=n)
        return genome

    def next_generation(self, fitnesses):
        new_inds = []
        n_elite = self.ind_num // 10

        elite_indices = np.argsort(fitnesses)[-n_elite:][::-1]

        for idx in elite_indices:
            elite = self.inds[idx].clone()
            new_inds.append(elite)

        while len(new_inds) < self.ind_num:
            p1, p2 = self.select_parents(fitnesses)
            g1, g2 = self.crossover(p1.genome, p2.genome)
            g1 = self.mutate(g1)
            g2 = self.mutate(g2)

            child1 = Individual(genome=g1)
            child2 = Individual(genome=g2)

            new_inds.extend([child1, child2])

        self.inds = new_inds[:self.ind_num]



torch.set_grad_enabled(False)
land_prey_num = 45
land_pred_num = 22
water_prey_num = 60
water_pred_num = 10

ga_land_prey = GeneticAlgorithm(nn_class=LandPreyNet, pop_size=land_prey_num, g1_size=water_prey_num, g2_size=land_pred_num, g3_size=water_pred_num)
ga_water_prey = GeneticAlgorithm(nn_class=WaterPreyNet, pop_size=water_prey_num, g1_size=land_prey_num, g2_size=land_pred_num, g3_size=water_pred_num)
ga_land_pred = GeneticAlgorithm(nn_class=LandPredNet, pop_size=land_pred_num, g1_size=water_pred_num, g2_size=land_prey_num, g3_size=water_prey_num)
ga_water_pred = GeneticAlgorithm(nn_class=WaterPredNet, pop_size=water_pred_num, g1_size=land_pred_num, g2_size=land_prey_num, g3_size=water_prey_num)

cycles = 5
gpc = 5

land_prey_args = {
    "age": 0,
    "energy": 30,
    "hydration": 30,
    "p_reproduce": 0.6,
    "max_energy": 200,
    "max_hydration": 200,
    "energy_from_food": 70,
    "hydration_from_water": 100,
    "vision_range": 4,
    "vision_angle": 180,
}

water_prey_args = {
    "age": 0,
    "energy": 30,
    "p_reproduce": 0.6,
    "max_energy": 200,
    "max_hydration": 200,
    "energy_from_food": 50,
    "hydration_from_water": 100,
    "vision_range": 4,
    "vision_angle": 180,
}

land_pred_args = {
    "age": 0,
    "energy": 30,
    "hydration": 30,
    "p_reproduce": 0.8,
    "max_energy": 200,
    "max_hydration": 200,
    "energy_from_food": 140,
    "hydration_from_water": 100,
    "vision_range": 8,
    "vision_angle": 90,
}

water_pred_args = {
    "age": 0,
    "energy": 30,
    "p_reproduce": 0.8,
    "max_energy": 200,
    "max_hydration": 200,
    "energy_from_food": 140,
    "hydration_from_water": 100,
    "vision_range": 8,
    "vision_angle": 90,
}

if os.path.exists("land_prey_net.pth"):

    delete_nns = input("Delete old NNs? Y/N\n")
    if delete_nns.lower() == "y":
        if os.path.exists("land_prey_net.pth"):
            os.remove("land_prey_net.pth")

        if os.path.exists("water_prey_net.pth"):
            os.remove("water_prey_net.pth")

        if os.path.exists("land_pred_net.pth"):
            os.remove("land_pred_net.pth")

        if os.path.exists("water_pred_net.pth"):
            os.remove("water_pred_net.pth")

for cyc in range(cycles):


    #---------------LAND CREATURES---------------------

    print(f"---- Land Prey Training: Cycle {cyc+1} ----")
    for gen in range(gpc):
        fitnesses = ga_land_prey.evaluate_fitness(PreyPred, LandPrey, WaterPrey, LandPredator, WaterPredator, land_prey_args, water_prey_args, land_pred_args, water_pred_args)
        print(f"Generation {gen+1} - Best fitness: {fitnesses.max()}")

        best_idx = np.argmax(fitnesses)
        best_ind = ga_land_prey.inds[best_idx]

        ga_land_prey.next_generation(fitnesses)

    # Save best
    best_nn = ga_land_prey.nn_class()
    torch.nn.utils.vector_to_parameters(
        torch.tensor(best_ind.genome, dtype=torch.float32),
        best_nn.parameters()
    )

    torch.save(best_nn.state_dict(), "land_prey_net.pth")
    del best_nn

    print("Saved land prey network.")



    print (f"---- Land Predator Training: Cycle {cyc+1} ----")
    for gen in range(gpc):
        fitnesses  = ga_land_pred.evaluate_fitness(PreyPred, LandPredator, WaterPredator, LandPrey, WaterPrey, land_pred_args, water_pred_args, land_prey_args, water_prey_args)
        print(f"Generation {gen+1} - Best fitness: {fitnesses.max()}")

        best_idx = np.argmax(fitnesses)
        best_ind = ga_land_pred.inds[best_idx]

        ga_land_pred.next_generation(fitnesses)

    # Save best
    best_nn = ga_land_pred.nn_class()
    torch.nn.utils.vector_to_parameters(
        torch.tensor(best_ind.genome, dtype=torch.float32),
        best_nn.parameters()
    )

    torch.save(best_nn.state_dict(), "land_pred_net.pth")
    del best_nn
    print("Saved land pred network.")



    #---------------WATER CREATURES--------------------
    print(f"---- Water Prey Training: Cycle {cyc+1} ----")
    for gen in range(gpc):
        fitnesses = ga_water_prey.evaluate_fitness(PreyPred, WaterPrey, LandPrey, LandPredator, WaterPredator, water_prey_args, land_prey_args, land_pred_args, water_pred_args)
        print(f"Generation {gen+1} - Best fitness: {fitnesses.max()}")

        best_idx = np.argmax(fitnesses)
        best_ind = ga_water_prey.inds[best_idx]

        ga_water_prey.next_generation(fitnesses)

    # Save best
    best_nn = ga_water_prey.nn_class()
    torch.nn.utils.vector_to_parameters(
        torch.tensor(best_ind.genome, dtype=torch.float32),
        best_nn.parameters()
    )

    torch.save(best_nn.state_dict(), "water_prey_net.pth")
    del best_nn
    print("Saved water prey network.")



    print (f"---- Water Predator Training: Cycle {cyc+1} ----")
    for gen in range(gpc):
        fitnesses  = ga_water_pred.evaluate_fitness(PreyPred, WaterPredator, LandPredator, LandPrey, WaterPrey, water_pred_args, land_pred_args, land_prey_args, water_prey_args)
        print(f"Generation {gen+1} - Best fitness: {fitnesses.max()}")

        best_idx = np.argmax(fitnesses)
        best_ind = ga_water_pred.inds[best_idx]

        ga_water_pred.next_generation(fitnesses)

    # Save best
    best_nn = ga_water_pred.nn_class()
    torch.nn.utils.vector_to_parameters(
        torch.tensor(best_ind.genome, dtype=torch.float32),
        best_nn.parameters()
    )

    torch.save(best_nn.state_dict(), "water_pred_net.pth")
    del best_nn
    print("Saved water pred network.")