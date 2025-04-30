import random

# Genetic Algorithm configuration
POPULATION_SIZE = 100
GENES = '''abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ 1234567890, .-;:_!"#%&/()=?@${[]}'''
TARGET = "I love GeeksforGeeks"

class Individual:
    """Class representing an individual in the population."""
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.calculate_fitness()

    @classmethod
    def mutated_gene(cls):
        """Generate a random gene."""
        return random.choice(GENES)

    @classmethod
    def create_gnome(cls):
        """Create a random chromosome (list of genes)."""
        return [cls.mutated_gene() for _ in range(len(TARGET))]

    def mate(self, partner):
        """Crossover with another individual to produce an offspring."""
        child_chromosome = []
        for gp1, gp2 in zip(self.chromosome, partner.chromosome):
            prob = random.random()
            if prob < 0.45:
                child_chromosome.append(gp1)
            elif prob < 0.90:
                child_chromosome.append(gp2)
            else:
                child_chromosome.append(self.mutated_gene())
        return Individual(child_chromosome)

    def calculate_fitness(self):
        """Fitness score: number of characters differing from target."""
        return sum(1 for ch, target_ch in zip(self.chromosome, TARGET) if ch != target_ch)

def main():
    generation = 1
    found = False

    # Create initial population
    population = [Individual(Individual.create_gnome()) for _ in range(POPULATION_SIZE)]

    while not found:
        # Sort population based on fitness
        population.sort(key=lambda ind: ind.fitness)

        # If perfect match is found
        if population[0].fitness == 0:
            found = True
            break

        # Display best of current generation
        print(f"Generation: {generation}\tString: {''.join(population[0].chromosome)}\tFitness: {population[0].fitness}")

        # Create new generation
        new_generation = []

        # Elitism: carry top 10% directly
        elite_size = POPULATION_SIZE // 10
        new_generation.extend(population[:elite_size])

        # Generate rest 90% through crossover and mutation
        for _ in range(POPULATION_SIZE - elite_size):
            parent1 = random.choice(population[:50])
            parent2 = random.choice(population[:50])
            child = parent1.mate(parent2)
            new_generation.append(child)

        population = new_generation
        generation += 1

    # Final result
    print(f"Generation: {generation}\tString: {''.join(population[0].chromosome)}\tFitness: {population[0].fitness}")

if __name__ == "__main__":
    main()
