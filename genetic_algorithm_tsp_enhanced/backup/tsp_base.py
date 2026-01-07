import pygame
from pygame.locals import *
import random
import itertools
from genetic_algorithm import mutate, order_crossover, generate_random_population, calculate_fitness, sort_population, default_problems
from draw_functions import draw_paths, draw_plot, draw_cities
import sys
import numpy as np
import pygame
from benchmark_att48 import *


# Define constant values
# pygame
WIDTH, HEIGHT = 800, 400
NODE_RADIUS = 10
FPS = 30
PLOT_X_OFFSET = 450

# GA
N_CITIES = 48
POPULATION_SIZE = 100 #### ---> Population initialization directly influences the efficiency and convergence of the process
N_GENERATIONS = None
MUTATION_PROBABILITY = 0.5 #### ---> Probability of a new individual, created from the crossing of the parents, undergoing mutation
MAX_EPOCHS = 100  #### ---> Maximum number of generations

# Define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)


# ----- Using att48 benchmark
WIDTH, HEIGHT = 1500, 800
att_cities_locations = np.array(att_48_cities_locations)
max_x = max(point[0] for point in att_cities_locations)
max_y = max(point[1] for point in att_cities_locations)
scale_x = (WIDTH - PLOT_X_OFFSET - NODE_RADIUS) / max_x
scale_y = HEIGHT / max_y
cities_locations = [(int(point[0] * scale_x + PLOT_X_OFFSET),
                      int(point[1] * scale_y)) for point in att_cities_locations]

# Benchmark target solution built in the same coordinate space as cities_locations
target_solution = [cities_locations[i-1] for i in att_48_cities_order]
fitness_target_solution = calculate_fitness(target_solution)
print(f"Benchmark fitness (att48 order): {round(fitness_target_solution, 2)}")
# ----- Using att48 benchmark


# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("TSP Solver using Pygame")
clock = pygame.time.Clock()
generation_counter = itertools.count(start=1)  # Start the counter at 1


# Create Initial Population
population = generate_random_population(cities_locations, POPULATION_SIZE)
best_fitness_values = []
best_solutions = []


# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_q:
                running = False

    generation = next(generation_counter)

    # Stop after MAX_EPOCHS generations
    if generation > MAX_EPOCHS:
        print(f"Reached max epochs ({MAX_EPOCHS}). Exiting.")
        running = False
        continue

    screen.fill(WHITE)

    population_fitness = [calculate_fitness(individual) for individual in population]

    population, population_fitness = sort_population(population, population_fitness)

    best_fitness = population_fitness[0]  # reuse computed fitness
    best_solution = population[0]

    best_fitness_values.append(best_fitness)
    best_solutions.append(best_solution)

    draw_plot(screen, list(range(len(best_fitness_values))), best_fitness_values, y_label="Fitness - Distance (pxls)")

    draw_cities(screen, cities_locations, RED, NODE_RADIUS)
    draw_paths(screen, best_solution, BLUE, width=3)
    draw_paths(screen, population[1], rgb_color=(128, 128, 128), width=1)

    print(f"Generation {generation}: Best fitness = {round(best_fitness, 2)}")

    new_population = [population[0]]  # Keep the best individual: ELITISM

    while len(new_population) < POPULATION_SIZE:
        # selection based on fitness probability
        probability = 1 / np.array(population_fitness)
        parent1, parent2 = random.choices(population, weights=probability, k=2)

        # If you want proper crossover between two parents, switch to (parent1, parent2)
        child1 = order_crossover(parent1, parent1)
        child1 = mutate(child1, MUTATION_PROBABILITY)

        new_population.append(child1)

    population = new_population

    pygame.display.flip()
    clock.tick(FPS)

# Post-processing: show final sequence and accuracy vs benchmark

def route_to_indices(route, cities_ref):
    """
    Convert a route (list of city coordinates) into indices relative to cities_ref.
    Returns a list of indices (0-based).
    """
    # Build a mapping from coordinate to index for O(1) lookups
    coord_to_index = {tuple(coord): idx for idx, coord in enumerate(cities_ref)}
    indices = []
    for city in route:
        idx = coord_to_index.get(tuple(city))
        if idx is None:
            # Fallback: find by equality scan (should not happen if references match)
            for j, ref in enumerate(cities_ref):
                if tuple(ref) == tuple(city):
                    idx = j
                    break
        if idx is None:
            raise ValueError("City in route not found in reference list.")
        indices.append(idx)
    return indices

if len(best_solutions) > 0:
    final_best = best_solutions[-1]
    final_fitness = best_fitness_values[-1]

    # Convert final best and benchmark to index sequences (0-based)
    final_indices_0 = route_to_indices(final_best, cities_locations)
    benchmark_indices_0 = [i - 1 for i in att_48_cities_order]  # convert to 0-based

    # Compute distance-wise accuracy
    dist_accuracy = (final_fitness / fitness_target_solution) * 100

    # Compute position-wise accuracy
    total_len = min(len(final_indices_0), len(benchmark_indices_0))
    matches = sum(1 for a, b in zip(final_indices_0[:total_len], benchmark_indices_0[:total_len]) if a == b)
    pos_accuracy = (matches / total_len) * 100 if total_len > 0 else 0.0

    # Prepare 1-based sequence for display
    final_indices_1 = [i + 1 for i in final_indices_0]

    print("\n=== Final Results ===")
    print(f"Generations run: {min(MAX_EPOCHS, len(best_fitness_values))}")
    print(f"Final best fitness: {round(final_fitness, 2)}")
    print(f"Benchmark fitness (att48 order): {round(fitness_target_solution, 2)}")
    print(f"Distance-wise accuracy vs benchmark: {dist_accuracy:.2f}%")   
    print(f"Final city sequence (1-based indices): {final_indices_1}")
    print(f"Benchmark city sequence (1-based indices): {att_48_cities_order}")
    print(f"Position-wise accuracy vs benchmark: {pos_accuracy:.2f}%")

#### exit software
# pygame.quit()
# sys.exit()