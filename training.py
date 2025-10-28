import numpy as np
import torch
import copy
import random
from agents import LandPrey, LandPredator, LandPreyNet, LandPredNet
from model import PreyPred
from mesa.experimental.devs import ABMSimulator

class Individual:
    def __init__(self, nn_class):
        self.nn_class = nn_class
        self.nn = nn_class()
        self.genome = self.get_genome()
        self.fitness = None

    def get_genome(self):
        return torch.nn.utils.parameters_to_vector(self.nn.parameters()).detach().cpu().numpy()

    def set_genome(self, genome):
        vec = torch.tensor(genome, dtype=torch.float32)
        torch.nn.utils.vector_to_parameters(vec, self.nn.parameters())
        self.genome = genome



class GeneticAlgorithm:
    def __init__(self, nn_class, pop_size=50, mut_rate=0.02, mut_strength=0.1):
        self.nn_class = nn_class
        self.pop_size = pop_size
        self.mut_rate = mut_rate
        self.mut_strength = mut_strength

        # Initialise neural nets with random weights
        self.population = [Individual(nn_class) for _ in range(pop_size)]

    def evaluate_fitness(self, model_class, training_class, opponent_class, training_args, opponent_args, steps=100):
        """Evaluate each genome by running a Mesa simulation."""
        fitnesses = []
        count = 0

        for ind in self.population:

            model = model_class(simulator=ABMSimulator())  # Reset model

            training_agent = training_class(nn=ind.nn, model = model, cell=random.choice(model.land_cells), **training_args)
            opponent_agent = opponent_class(model=model, cell=random.choice(model.land_cells), **opponent_args)

        for _ in range(steps):
            model.step()

        # Fitness - reproductions + age
        ind.fitness = training_agent.rep_count + training_agent.age
        fitnesses.append(ind.fitness)

        print("Prey", count, "fitness:", ind.fitness)
        count += 1

        return np.array(fitnesses)


    def select_parents(self, fitnesses):
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
        elite_idx = np.argmax(fitnesses)
        new_population.append(copy.deepcopy(self.population[elite_idx]))

        while len(new_population) < self.pop_size:
            p1, p2 = self.select_parents(fitnesses)
            g1, g2 = self.crossover(p1.genome, p2.genome)
            g1 = self.mutate(g1)
            g2 = self.mutate(g2)

            child1 = Individual(self.nn_class)
            child1.set_genome(g1)
            child2 = Individual(self.nn_class)
            child2.set_genome(g2)

            new_population.extend([child1, child2])

        self.population = new_population[:self.pop_size]



ga = GeneticAlgorithm(nn_class=LandPreyNet)

generations = 20

training_args = {
    "age": 0,
    "rep_count": 0,
    "energy": 50,
    "hydration": 50,
    "p_reproduce": 0.5,
    "max_energy": 100,
    "max_hydration": 100,
    "energy_from_food": 4,
    "hydration_from_water": 20,
    "vision_range": 4,
    "vision_angle": 180,
}

opponent_args = {
    "age": 0,
    "rep_count": 0,
    "energy": 50,
    "hydration": 50,
    "p_reproduce": 0.5,
    "max_energy": 100,
    "max_hydration": 100,
    "energy_from_food": 10,
    "hydration_from_water": 20,
    "vision_range": 8,
    "vision_angle": 90,
}

for gen in range(generations):
    fitnesses = ga.evaluate_fitness(PreyPred, LandPrey, LandPredator, training_args, opponent_args)
    print(f"Generation {gen} - Best fitness: {fitnesses.max()}")
    ga.next_generation(fitnesses)


# Save best
best_idx = np.argmax([ind.fitness for ind in ga.population])
best_individual = ga.population[best_idx]

torch.save(best_individual.nn.state_dict(), "best_prey_net.pth")
print("Saved best neural network.")
