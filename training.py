import numpy as np
import torch
import copy
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
    def __init__(self, nn_class, pop_size=20, opp_size=5, mut_rate=0.05, mut_strength=0.2):
        self.nn_class = nn_class
        self.pop_size = pop_size
        self.opp_size = opp_size
        self.mut_rate = mut_rate
        self.mut_strength = mut_strength

        # Initialise neural nets with random weights
        self.population = [Individual(nn_class) for _ in range(pop_size)]



    def evaluate_fitness(self, model_class, training_class, opponent_class, training_args, opponent_args, steps=100):
        """Evaluate each genome by running a Mesa simulation."""
        fitnesses = []

        model = model_class(simulator=ABMSimulator(), initial_land_prey=0, initial_land_pred=0, data_collect = False, seed=random.randint(0, 9999))  # Reset model

        for ind in self.population:
            training_agent = training_class(nn=ind.nn, model = model, cell=random.choice(model.land_cells), **training_args)
            ind.agent = training_agent
            
        for i in range(self.opp_size):
            opponent_agent = opponent_class(model=model, cell=random.choice(model.land_cells), **opponent_args)

        for s in range(steps):
            model.step()


        for ind in self.population:

            if ind.agent.energy <= 1:
                death_penalty = -10
            else:
                death_penalty = 0

            #Prey fitness - reproductions + amount drank + moves + fleeing efficiency - death penalty
            if training_class == LandPrey:
                flee_eff = ind.agent.pred_flees / max(1, ind.agent.moves)
                ind.fitness = 5 * ind.agent.rep_count + 0.5 * ind.agent.drank + 0.1 * ind.agent.moves + 0.5 * flee_eff + death_penalty

            #Predator fitness - reproductions + amount drank + prey eaten + moves + hunt efficiency - death penalty
            elif training_class == LandPredator:
                hunt_eff = ind.agent.prey_chases / max(1, ind.agent.moves)
                ind.fitness = 5 * ind.agent.rep_count + 0.5 * ind.agent.drank + 3 * ind.agent.prey_eaten + 0.1 * ind.agent.moves + 0.5 * hunt_eff + death_penalty

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



ga_prey = GeneticAlgorithm(nn_class=LandPreyNet, pop_size=90, opp_size=15)
ga_pred = GeneticAlgorithm(nn_class=LandPredNet, pop_size=15, opp_size=90)

cycles = 50
generations = 5

prey_args = {
    "age": 0,
    "energy": 100,
    "hydration": 100,
    "p_reproduce": 0.3,
    "max_energy": 120,
    "max_hydration": 120,
    "energy_from_food": 5,
    "hydration_from_water": 20,
    "vision_range": 4,
    "vision_angle": 180,
}

pred_args = {
    "age": 0,
    "energy": 100,
    "hydration": 100,
    "p_reproduce": 0.4,
    "max_energy": 100,
    "max_hydration": 100,
    "energy_from_food": 15,
    "hydration_from_water": 20,
    "vision_range": 8,
    "vision_angle": 90,
}
for cyc in range(cycles):

    print(f"---- Prey Training: Cycle {cyc}----")
    for gen in range(generations):
        fitnesses = ga_prey.evaluate_fitness(PreyPred, LandPrey, LandPredator, prey_args, pred_args)
        print(f"Generation {gen} - Best fitness: {fitnesses.max()}")

        best_idx = np.argmax(fitnesses)
        best_ind = ga_prey.population[best_idx]

        ga_prey.next_generation(fitnesses)

    # Save best
    torch.save(best_ind.nn.state_dict(), "prey_net.pth")
    print("Saved prey network.")


    print (f"---- Predator Training: Cycle {cyc} ----")
    for gen in range(generations):
        fitnesses = ga_pred.evaluate_fitness(PreyPred, LandPredator, LandPrey, pred_args, prey_args)
        print(f"Generation {gen} - Best fitness: {fitnesses.max()}")

        best_idx = np.argmax(fitnesses)
        best_ind = ga_pred.population[best_idx]

        ga_pred.next_generation(fitnesses)

    # Save best
    torch.save(best_ind.nn.state_dict(), "pred_net.pth")
    print("Saved pred network.")
