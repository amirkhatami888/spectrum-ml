import numpy as np
import tensorflow as tf
from geneticalgorithm2 import geneticalgorithm2 as ga

# Define parameters for the genetic algorithm
POPULATION_SIZE = 2
NUM_GENERATIONS = 2
MAX_HIDDEN_LAYERS = 3
MIN_HIDDEN_LAYERS = 1
MAX_NEURONS_PER_LAYER = 2
INPUT_SIZE = 5  # Number of features in each time step
OUTPUT_SIZE = INPUT_SIZE  # Assuming same number of features in output
activation_list = ['relu', 'selu', 'linear']

# Define fitness function (assuming MSE loss for time series prediction)
def fitness_function(chromosome):
  # Extract number of hidden layers
  num_hidden_layers = np.clip(int(chromosome[0]), MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS)
  # Extract neurons per layer
  neurons_per_layer = [int(chromosome[i]) for i in range(1, num_hidden_layers+1)]
  # Extract activations
  activations = [activation_list[int(chromosome[i])] for i in range(num_hidden_layers+1, 2*num_hidden_layers+1)]

  # Build neural network (using standard ANN)
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(neurons_per_layer[0], activation=activations[0], input_shape=(INPUT_SIZE,)))
  for i in range(1, num_hidden_layers):
    model.add(tf.keras.layers.Dense(neurons_per_layer[i], activation=activations[i]))
  model.add(tf.keras.layers.Dense(OUTPUT_SIZE))  # Final layer predicts output time series

  # Compile model
  model.compile(optimizer='adam', loss='mse', metrics=['mae'])  # Consider using MAE for time series

  # Train model (using example time series data)
  # Replace with your actual time series data (X_train, y_train)
  X_train = np.random.rand(100, INPUT_SIZE)  # Sample time series data (adjust as needed)
  y_train = X_train  # Assuming output is same time series

  model.fit(X_train, y_train, epochs=3, verbose=0)

  # Evaluate model fitness (consider using validation data for a more robust evaluation)
  dummy_X = np.random.rand(10, INPUT_SIZE)  # Sample data for evaluation
  dummy_y = np.random.rand(10, INPUT_SIZE)
  mse, _ = model.evaluate(dummy_X, dummy_y, verbose=0)

  return -mse  # Minimize mean squared error

# Define variable boundaries (including number of hidden layers)
varbound = np.array([[MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS]] +
                    [[1, MAX_NEURONS_PER_LAYER]] * MAX_HIDDEN_LAYERS +
                    [[0, len(activation_list)-1]] * MAX_HIDDEN_LAYERS)

# Create genetic algorithm optimizer
model = ga(function=fitness_function, dimension=2*MAX_HIDDEN_LAYERS+1,
             variable_type='int', variable_boundaries=varbound)

# Run the optimization
model.run(POPULATION_SIZE, NUM_GENERATIONS)

# Print best solution found
best_solution = model.output_dict['variable']
print("Best solution:", best_solution)
print("Fitness:", -model.output_dict['function'])  # Un-negate MSE for readability

# Extract and print neural network structure (considering dynamically determined layers)
num_hidden_layers = best_solution[0]
neurons_per_layer = [best_solution[i] for i in range(1, num_hidden_layers+1)]
activations = [activation_list[int(chromosome[i])] for i in range(num_hidden_layers+1, 2*num_hidden_layers+1)]
print("Neural Network Structure:", best_solution)
print("Number of Hidden Layers:", num_hidden_layers)
print("Neurons per Layer:", neurons_per_layer)
print("Activation Functions:", activations)
