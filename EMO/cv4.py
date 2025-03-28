import random
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def random_alpha(size):
    return ''.join(random.choice('01') for _ in range(size))

def gamma(alpha, k, n):
    return [gray_dec(alpha[i*k:(i+1)*k]) for i in range(n)]

def gray_dec(alpha):
    
    binary = int(alpha[0])
    result = binary
    for bit in alpha[1:]:
        binary ^= int(bit)
        result = (result << 1) | binary
    return result

def leaky_relu(x):
    return max(0.01*x, x)

def fitness(solution, values, weights, capacity):
    solution_array = np.array([int(bit) for bit in solution])
    total_value = solution_array @ values
    total_weight = solution_array @ weights
    
    value_fitness = leaky_relu(total_value)
    weight_penalty = leaky_relu(capacity - total_weight)
    
    return value_fitness * weight_penalty


def generate_population(size, n):
    return [random_alpha(n) for _ in range(size)]

def selection(population, fitness_values):
    return random.choices(population, weights=fitness_values, k=len(population))

def crossover(parent1, parent2):
    return parent1[:len(parent1)//2] + parent2[len(parent2)//2:]

def mutation(solution):
    return ''.join(random.choice('01') if random.random() < MUTATION_RATE else solution[i] for i in range(len(solution)))

#1 initial population
#2 selection
#3 crossover
#4 mutation
#5 loop 2-3-4
#6 return best solution

POPULATION_SIZE = 100

ITERATIONS = 1000

MUTATION_RATE = 0.05

knapsack_max_weight = 20
knapsack_values = [4, 2, 10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]
knapsack_weights = [12, 1, 4, 2, 1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25]

N = len(knapsack_values)


def main():
    population = generate_population(POPULATION_SIZE, N)
    for _ in range(ITERATIONS):
        fitness_values = [fitness(solution, knapsack_values, knapsack_weights, knapsack_max_weight) for solution in population]
        
        parents = selection(population, fitness_values)
        
        offspring = []
        for i in range(0, len(parents)-1, 2):
            offspring.append(crossover(parents[i], parents[i+1]))
            offspring.append(crossover(parents[i+1], parents[i]))
        
        offspring = [mutation(solution) for solution in offspring]
        
        population = offspring[:]

        print(f"Iteration {_}, Best solution: {max(population, key=lambda x: fitness(x, knapsack_values, knapsack_weights, knapsack_max_weight))}, fitness: {fitness(max(population, key=lambda x: fitness(x, knapsack_values, knapsack_weights, knapsack_max_weight)), knapsack_values, knapsack_weights, knapsack_max_weight)}")

    best_solution = max(population, key=lambda x: fitness(x, knapsack_values, knapsack_weights, knapsack_max_weight))
    print("Best solution:", best_solution)
    print("Fitness value:", fitness(best_solution, knapsack_values, knapsack_weights, knapsack_max_weight))

    chosen_items = [i+1 for i, bit in enumerate(best_solution) if bit == '1']
    print(f"Items chosen: {chosen_items}")
    print(f"Total weight: {sum(knapsack_weights[i] for i in range(len(best_solution)) if best_solution[i] == '1')}")

def create_correlation_matrix():
    # Create random data
    np.random.seed(42)
    data = {
        'Age': np.random.normal(30, 10, 100),
        'Income': np.random.normal(50000, 15000, 100),
        'Experience': np.random.normal(8, 4, 100),
        'Performance': np.random.normal(85, 10, 100),
        'Satisfaction': np.random.normal(7.5, 1.5, 100)
    }
    
    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Correlation Matrix')
    plt.tight_layout()
    plt.show()
    
    # Print correlation matrix
    print("\nCorrelation Matrix:")
    print(corr_matrix)

if __name__ == "__main__":
    main()
    create_correlation_matrix()

