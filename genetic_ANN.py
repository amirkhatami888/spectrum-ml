import numpy as np
import tensorflow as tf
from geneticalgorithm2 import geneticalgorithm2 as ga

# Define parameters for the genetic algorithm
POPULATION_SIZE = 2
NUM_GENERATIONS = 2
NUM_HIDDEN_LAYERS = 3
MAX_NEURONS_PER_LAYER = 2
INPUT_SIZE = 5
OUTPUT_SIZE = 1
activation_list = ['relu', 'selu', 'linear']

# Define fitness function
def fitness_function(chromosome):
    num_hidden_layers = int(chromosome[0])
    neurons_per_layer = [int(chromosome[i]) for i in range(1, NUM_HIDDEN_LAYERS+1)]
    activations = [activation_list[int(chromosome[i])] for i in range(NUM_HIDDEN_LAYERS+1, 2*NUM_HIDDEN_LAYERS+1)]
    
    # Build neural network
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(INPUT_SIZE,)))
    for i in range(num_hidden_layers):
        model.add(tf.keras.layers.Dense(neurons_per_layer[i], activation=activations[i]))
    model.add(tf.keras.layers.Dense(OUTPUT_SIZE, activation='linear'))
    
    # Compile model
    model.compile(optimizer='adam', loss='mse', metrics=['accuracy'])
    
    # Train model (using dummy data for illustration)
    dummy_X = np.random.rand(100, INPUT_SIZE)
    dummy_y = np.random.randint(0, OUTPUT_SIZE, size=(100,))
    
    model.fit(dummy_X, dummy_y, epochs=3, verbose=0)
    
    # Evaluate model fitness
    _, accuracy = model.evaluate(dummy_X, dummy_y, verbose=0)
    
    return accuracy
varbound = np.array([[1, 1]] + [[1, MAX_NEURONS_PER_LAYER]] * NUM_HIDDEN_LAYERS +
                   [[0, len(activation_list)-1]] * NUM_HIDDEN_LAYERS)

# Create genetic algorithm optimizer
model = ga(function=fitness_function, dimension=2*NUM_HIDDEN_LAYERS+1, variable_type='int', variable_boundaries=varbound)

#run the optimization
model.run(POPULATION_SIZE, NUM_GENERATIONS)

# Print best solution found
best_solution = model.output_dict['variable']
print("Best solution:", best_solution)
print("Fitness:", model.output_dict['function'])

# Extract and print neural network structure
num_hidden_layers = int(best_solution[0])
neurons_per_layer = [int(best_solution[i]) for i in range(1, NUM_HIDDEN_LAYERS+1)]
activations = [activation_list[int(best_solution[i])] for i in range(NUM_HIDDEN_LAYERS+1, 2*NUM_HIDDEN_LAYERS+1)]
print("Neural Network Structure:", best_solution)
print("Number of Hidden Layers:", num_hidden_layers)
print("Neurons per Layer:", neurons_per_layer)
print("Activation Functions:", activations)