# main.py
import random
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

class City:
    """Represents a city with coordinates."""
    def __init__(self, x: float, y: float, name: str = ""):
        self.x = x
        self.y = y
        self.name = name or f"City({x},{y})"
    
    def distance_to(self, other: 'City') -> float:
        """Calculate Euclidean distance to another city."""
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def __repr__(self):
        return self.name

class Route:
    """Represents a route through all cities."""
    def __init__(self, cities: List[City], order: Optional[List[int]] = None):
        self.cities = cities
        if order is None:
            # Random initial order (excluding start/end city which should be same)
            self.order = list(range(len(cities)))
            random.shuffle(self.order)
        else:
            self.order = order
    
    def get_distance(self) -> float:
        """Calculate total distance of the route."""
        total_distance = 0
        for i in range(len(self.order)):
            from_city = self.cities[self.order[i]]
            to_city = self.cities[self.order[(i + 1) % len(self.order)]]
            total_distance += from_city.distance_to(to_city)
        return total_distance
    
    def get_fitness(self) -> float:
        """Fitness is inverse of distance (shorter routes are better)."""
        return 1 / self.get_distance()
    
    def __repr__(self):
        return f"Route: {' -> '.join(str(self.cities[i]) for i in self.order)} -> {self.cities[self.order[0]]}"

class GeneticAlgorithm:
    """Genetic Algorithm for solving TSP."""
    def __init__(self, cities: List[City], population_size: int = 100, elite_size: int = 20,
                 mutation_rate: float = 0.01, generations: int = 500):
        self.cities = cities
        self.population_size = population_size
        self.elite_size = elite_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.best_distances = []
        
    def create_initial_population(self) -> List[Route]:
        """Create initial population of random routes."""
        return [Route(self.cities) for _ in range(self.population_size)]
    
    def rank_routes(self, population: List[Route]) -> List[int]:
        """Rank routes by fitness."""
        fitness_results = []
        for i, route in enumerate(population):
            fitness_results.append((i, route.get_fitness()))
        
        # Sort by fitness in descending order
        fitness_results.sort(key=lambda x: x[1], reverse=True)
        return [x[0] for x in fitness_results]
    
    def selection(self, population: List[Route], ranked_routes: List[int]) -> List[Route]:
        """Select best routes for mating pool."""
        selection_results = []
        
        # Always include elite routes
        for i in range(self.elite_size):
            selection_results.append(population[ranked_routes[i]])
        
        # Roulette wheel selection for remaining spots
        fitness_sum = sum(population[i].get_fitness() for i in ranked_routes)
        cum_sum = 0
        fitness_cumsum = []
        
        for i in ranked_routes:
            cum_sum += population[i].get_fitness()
            fitness_cumsum.append(cum_sum / fitness_sum)
        
        for i in range(self.elite_size, self.population_size):
            pick = random.random()
            for j, cumulative_fitness in enumerate(fitness_cumsum):
                if pick <= cumulative_fitness:
                    selection_results.append(population[ranked_routes[j]])
                    break
        
        return selection_results
    
    def breed(self, parent1: Route, parent2: Route) -> Route:
        """Create child route through ordered crossover."""
        gene_a = random.randint(0, len(parent1.order) - 1)
        gene_b = random.randint(0, len(parent1.order) - 1)
        
        start_gene = min(gene_a, gene_b)
        end_gene = max(gene_a, gene_b)
        
        child_order = []
        for i in range(start_gene, end_gene):
            child_order.append(parent1.order[i])
        
        remaining_cities = [city for city in parent2.order if city not in child_order]
        
        # Fill in remaining cities in order from parent2
        child_order = remaining_cities[:start_gene] + child_order + remaining_cities[start_gene:]
        
        return Route(self.cities, child_order)
    
    def breed_population(self, mating_pool: List[Route]) -> List[Route]:
        """Breed entire mating pool."""
        children = []
        
        # Keep elite routes untouched
        for i in range(self.elite_size):
            children.append(mating_pool[i])
        
        # Fill rest with bred children
        pool = random.sample(mating_pool, len(mating_pool))
        
        for i in range(self.elite_size, self.population_size):
            parent1 = pool[i % len(pool)]
            parent2 = pool[(i + 1) % len(pool)]
            child = self.breed(parent1, parent2)
            children.append(child)
        
        return children
    
    def mutate(self, route: Route) -> Route:
        """Mutate route by swapping two cities."""
        for i in range(len(route.order)):
            if random.random() < self.mutation_rate:
                j = random.randint(0, len(route.order) - 1)
                route.order[i], route.order[j] = route.order[j], route.order[i]
        
        return route
    
    def mutate_population(self, population: List[Route]) -> List[Route]:
        """Apply mutation to entire population."""
        mutated_population = []
        
        for i in range(self.population_size):
            if i < self.elite_size:
                mutated_population.append(population[i])  # Don't mutate elite
            else:
                mutated_population.append(self.mutate(population[i]))
        
        return mutated_population
    
    def next_generation(self, current_generation: List[Route]) -> List[Route]:
        """Create next generation."""
        ranked_routes = self.rank_routes(current_generation)
        selection_results = self.selection(current_generation, ranked_routes)
        children = self.breed_population(selection_results)
        next_gen = self.mutate_population(children)
        return next_gen
    
    def run(self) -> Tuple[Route, List[float]]:
        """Run the genetic algorithm."""
        population = self.create_initial_population()
        
        for i in range(self.generations):
            population = self.next_generation(population)
            best_route_idx = self.rank_routes(population)[0]
            best_distance = population[best_route_idx].get_distance()
            self.best_distances.append(best_distance)
            
            if i % 100 == 0:
                print(f"Generation {i}: Best distance = {best_distance:.2f}")
        
        # Return best route from final population
        best_route_idx = self.rank_routes(population)[0]
        best_route = population[best_route_idx]
        
        return best_route, self.best_distances

def plot_route(route: Route, title: str = "Best TSP Route"):
    """Plot the route on a 2D graph."""
    plt.figure(figsize=(10, 8))
    
    # Plot cities
    x_coords = [city.x for city in route.cities]
    y_coords = [city.y for city in route.cities]
    plt.scatter(x_coords, y_coords, c='red', s=100)
    
    # Label cities
    for i, city in enumerate(route.cities):
        plt.annotate(city.name, (city.x, city.y), xytext=(5, 5), textcoords='offset points')
    
    # Plot route
    for i in range(len(route.order)):
        from_city = route.cities[route.order[i]]
        to_city = route.cities[route.order[(i + 1) % len(route.order)]]
        plt.plot([from_city.x, to_city.x], [from_city.y, to_city.y], 'b-', alpha=0.7)
    
    plt.title(f"{title} (Distance: {route.get_distance():.2f})")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def plot_progress(best_distances: List[float]):
    """Plot the progress of the algorithm."""
    plt.figure(figsize=(10, 6))
    plt.plot(best_distances)
    plt.title("Algorithm Progress")
    plt.xlabel("Generation")
    plt.ylabel("Best Distance")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def main():
    # Create a set of cities
    city_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]
    cities = []
    
    for i, name in enumerate(city_names):
        x = random.uniform(0, 100)
        y = random.uniform(0, 100)
        cities.append(City(x, y, name))
    
    # Initialize and run the genetic algorithm
    ga = GeneticAlgorithm(
        cities=cities,
        population_size=100,
        elite_size=20,
        mutation_rate=0.01,
        generations=500
    )
    
    print("Starting genetic algorithm for TSP...")
    best_route, best_distances = ga.run()
    
    print(f"\nBest route found: {best_route}")
    print(f"Total distance: {best_route.get_distance():.2f}")
    
    # Plot results
    plot_route(best_route)
    plot_progress(best_distances)

if __name__ == "__main__":
    main()