import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import random
from agents import LandPrey, LandPredator, LandPreyNet, LandPredNet
from model import PreyPred
from mesa.experimental.devs import ABMSimulator

class Individual:
    def __init__(self, nn_class, genome=None):
        self.nn_class = nn_class
        self.nn = nn_class()
        if genome is not None:
            self.genome = self.set_genome(genome)
        self.genome = self.get_genome()
        self.fitness = None
        self.agent = None

    def get_genome(self):
        return torch.nn.utils.parameters_to_vector(self.nn.parameters()).detach().cpu().numpy()

    def set_genome(self, genome):
        vec = torch.tensor(genome, dtype=torch.float32)
        torch.nn.utils.vector_to_parameters(vec, self.nn.parameters())
        self.genome = genome

    def clone(self):
        return Individual(self.nn_class, genome=self.genome.copy())



class GeneticAlgorithm:
    def __init__(self, nn_class, pop_size=20, opp_size=5, mut_rate=0.075, mut_strength=0.4):
        self.nn_class = nn_class
        self.pop_size = pop_size
        self.opp_size = opp_size
        self.mut_rate = mut_rate
        self.mut_strength = mut_strength

        # Initialise neural nets with random weights
        self.population = [Individual(nn_class) for _ in range(pop_size)]



    def evaluate_fitness(self, model_class, training_class, opponent_class, training_args, opponent_args, steps=300):
        """Evaluate each genome by running a Mesa simulation."""
        fitnesses = []

        model = model_class(simulator=ABMSimulator(), initial_land_prey=0, initial_land_pred=0, data_collect = False, seed=random.randint(1, 9999))  # Reset model

        for ind in self.population:
            training_agent = training_class(nn=ind.nn, model = model, cell=random.choice(model.land_cells), **training_args)
            ind.agent = training_agent
            
        for i in range(self.opp_size):
            opponent_agent = opponent_class(model=model, cell=random.choice(model.land_cells), **opponent_args)

        for s in range(steps):
            model.step()


        for ind in self.population:

            if ind.agent.energy <= 1:
                death_penalty = -5
            else:
                death_penalty = 0

            #Prey fitness - survival + eating/drinking + reproduction + foraging + fleeing - death
            if training_class == LandPrey:
                foraging = ind.agent.food_approach + ind.agent.water_approach

                ind.fitness = (2 * ind.agent.age +
                               3 * ind.agent.food_eaten +
                               ind.agent.drank +
                               3 * ind.agent.rep_count +
                               2 * foraging +
                               ind.agent.pred_flees +
                               death_penalty)
                
            #Predator fitness - survival + eating/drinking + reproduction + moving + chasing  - death
            elif training_class == LandPredator:
                ind.fitness = (4 * ind.agent.age +
                               6 * ind.agent.prey_eaten +
                               3 * ind.agent.drank +
                               3 * ind.agent.rep_count +
                               3 * ind.agent.prey_chases +
                               death_penalty)
            
            fitnesses.append(ind.fitness)

        return np.array(fitnesses)



    def select_parents(self, fitnesses):
        
        min_fit = np.min(fitnesses)
        if min_fit < 0:
            fitnesses = fitnesses - min_fit + 1e-6  # shift up slightly

        probs = fitnesses / np.sum(fitnesses)
        idxs = np.random.choice(len(self.population), size=2, p=probs)
        return self.population[idxs[0]], self.population[idxs[1]]

    def crossover(self, genome1, genome2):
        point = np.random.randint(0, len(genome1))
        child1 = np.concatenate([genome1[:point], genome2[point:]])
        child2 = np.concatenate([genome2[:point], genome1[point:]])
        return child1, child2

    def mutate(self, genome):
        mask = np.random.rand(len(genome)) < self.mut_rate
        genome[mask] += np.random.normal(0, self.mut_strength, size=mask.sum())
        return genome

    def next_generation(self, fitnesses):
        new_population = []
        n_elite = self.pop_size // 10

        elite_indices = np.argsort(fitnesses)[-n_elite:][::-1]

        for idx in elite_indices:
            elite = self.population[idx].clone()
            new_population.append(elite)

        while len(new_population) < self.pop_size:
            p1, p2 = self.select_parents(fitnesses)
            g1, g2 = self.crossover(p1.genome, p2.genome)
            g1 = self.mutate(g1)
            g2 = self.mutate(g2)

            child1 = Individual(self.nn_class, genome=g1)
            child2 = Individual(self.nn_class, genome=g2)

            new_population.extend([child1, child2])

        self.population = new_population[:self.pop_size]



ga_prey = GeneticAlgorithm(nn_class=LandPreyNet, pop_size=100, opp_size=20)
ga_pred = GeneticAlgorithm(nn_class=LandPredNet, pop_size=20, opp_size=100)

cycles = 10
gpc = 10

prey_args = {
    "age": 0,
    "energy": 30,
    "hydration": 30,
    "p_reproduce": 0.15,
    "max_energy": 200,
    "max_hydration": 200,
    "energy_from_food": 70,
    "hydration_from_water": 100,
    "vision_range": 6,
    "vision_angle": 180,
}

pred_args = {
    "age": 0,
    "energy": 30,
    "hydration": 30,
    "p_reproduce": 0.2,
    "max_energy": 200,
    "max_hydration": 200,
    "energy_from_food": 140,
    "hydration_from_water": 100,
    "vision_range": 12,
    "vision_angle": 90,
}

if os.path.exists("prey_net.pth"):
    os.remove("prey_net.pth")

if os.path.exists("pred_net.pth"):
    os.remove("pred_net.pth")


for cyc in range(cycles):

    print(f"---- Prey Training: Cycle {cyc}----")
    for gen in range(gpc):
        fitnesses = ga_prey.evaluate_fitness(PreyPred, LandPrey, LandPredator, prey_args, pred_args)
        print(f"Generation {gen} - Best fitness: {fitnesses.max()}")

        best_idx = np.argmax(fitnesses)
        best_ind = ga_prey.population[best_idx]

        ga_prey.next_generation(fitnesses)

    # Save best
    torch.save(best_ind.nn.state_dict(), "prey_net.pth")
    print("Saved prey network.")


    print (f"---- Predator Training: Cycle {cyc} ----")
    for gen in range(gpc):
        fitnesses  = ga_pred.evaluate_fitness(PreyPred, LandPredator, LandPrey, pred_args, prey_args)
        print(f"Generation {gen} - Best fitness: {fitnesses.max()}")

        best_idx = np.argmax(fitnesses)
        best_ind = ga_pred.population[best_idx]

        ga_pred.next_generation(fitnesses)

    # Save best
    torch.save(best_ind.nn.state_dict(), "pred_net.pth")
    print("Saved pred network.")