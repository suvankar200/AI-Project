# Traveling Salesman Problem with Genetic Algorithm

This project implements a genetic algorithm to solve the Traveling Salesman Problem (TSP). The TSP asks: given a list of cities and the distances between them, what is the shortest possible route that visits each city exactly once and returns to the starting city?

## Project Structure

```
tsp_genetic_algorithm/
│
├── main.py              # Main implementation file
├── requirements.txt     # Project dependencies
├── README.md           # This file
└── examples/           # Example outputs (optional)
```

## Features

- Object-oriented implementation with clear class hierarchy
- Configurable genetic algorithm parameters
- Visualization of best route and algorithm progress
- Flexible input for different cities and coordinates

## Classes

1. **City**: Represents a city with x,y coordinates
2. **Route**: Represents a path through all cities
3. **GeneticAlgorithm**: Implements the genetic algorithm

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/tsp-genetic-algorithm.git
   cd tsp-genetic-algorithm
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

- Python 3.7+
- NumPy
- Matplotlib

## Usage

Run the main program:
```bash
python main.py
```

To customize the algorithm parameters, modify the `main()` function:
```python
ga = GeneticAlgorithm(
    cities=cities,
    population_size=100,  # Number of routes in each generation
    elite_size=20,        # Number of best routes to preserve
    mutation_rate=0.01,   # Probability of mutation
    generations=500       # Number of generations to run
)
```

## Algorithm Overview

1. **Initial Population**: Creates random routes through all cities
2. **Fitness Evaluation**: Calculates route distances (shorter is better)
3. **Selection**: Chooses best routes for breeding
4. **Crossover**: Creates new routes by combining parent routes
5. **Mutation**: Randomly swaps cities in some routes to maintain diversity
6. **Iteration**: Repeats steps 2-5 for the specified number of generations

## Output

The program generates two plots:
1. Best route found (cities connected by lines)
2. Algorithm progress (best distance vs. generation)

It also prints progress updates and the final best route.

## Future Improvements

- Add support for different selection methods (tournament, stochastic)
- Implement different crossover operators
- Add option to read cities from a file
- Parallel processing for large populations
- Interactive visualization
- Support for more complex constraints (time windows, capacity)

## Contributing

Feel free to open issues or submit pull requests with improvements.

## License

MIT License

