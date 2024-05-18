import numpy as np
import pandas as pd
import tensorflow as tf
from geneticalgorithm2 import geneticalgorithm2 as ga
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import keras

# Ensure TensorFlow is in Eager execution mode
tf.config.run_functions_eagerly(True)

# Create a simple dataset for test code
num_data = 10
num_features_input = 4
num_features_output = 3
data_input = np.random.rand(num_data, num_features_input)
data_output = np.random.rand(num_data, num_features_output)
pd.DataFrame(data_input).to_csv('inputdata.csv', index=False)
pd.DataFrame(data_output).to_csv('outputdata.csv', index=False)

# Define path
INPUT_path = r'inputdata.csv'
OUTPUT_path = r'outputdata.csv'

# Define data
INPUT_DATA = pd.read_csv(INPUT_path).values 
OUTPUT_DATA = pd.read_csv(OUTPUT_path).values

# Data type conversion
INPUT_DATA = INPUT_DATA.astype('float32')
OUTPUT_DATA = OUTPUT_DATA.astype('float32')

# Define data information
INPUT_SIZE_input = INPUT_DATA.shape[1]  # Number of features of input data
OUTPUT_SIZE_input = OUTPUT_DATA.shape[1]  # Number of features of output data
NUM_DATA = INPUT_DATA.shape[0]  # Number of data samples

# Normalization (assuming you want to normalize)
scaler = MinMaxScaler(feature_range=(0, 1))
INPUT_DATA_Normal = scaler.fit_transform(INPUT_DATA)
OUTPUT_DATA_Normal = scaler.fit_transform(OUTPUT_DATA)

# Split data into training and testing sets
test_size = 0.1
X_train, X_test, y_train, y_test = train_test_split(INPUT_DATA_Normal, OUTPUT_DATA_Normal, test_size=test_size, random_state=42)

# Define parameters for the genetic algorithm
max_num_iteration = 3
population_size = 5
mutation_probability = 0.1
elit_ratio = 0.01
crossover_probability = 0.5
parents_portion = 0.5
crossover_type = 'uniform'

# Define parameters for neural network
MAX_HIDDEN_LAYERS = 3  # Increased for more complex models
MIN_HIDDEN_LAYERS = 2  # Ensure at least one hidden layer
MAX_NEURONS_PER_LAYER = 15  # Increase for more complex models
MIN_NEURONS_PER_LAYER = 5  # Adjust for complexity
learning_rate = 0.01
NUM_EPOCHS = 10
BATCH_SIZE = 10
VERBOSE = 1
INPUT_SIZE = INPUT_SIZE_input
OUTPUT_SIZE = OUTPUT_SIZE_input
activation_list = ['relu', 'selu', 'linear']

# Define fitness function (assuming MSE loss for time series prediction)
def fitness_function(chromosome):
    # Extract number of hidden layers
    num_hidden_layers = np.clip(int(chromosome[0]), MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS)

    # Extract neurons per layer
    neurons_per_layer = [int(chromosome[i]) for i in range(1, num_hidden_layers + 1)]
    
    # Extract activations
    activations = [activation_list[int(chromosome[MAX_HIDDEN_LAYERS + i])] for i in range(1, num_hidden_layers + 1)]
    print(chromosome)
    print('Number of hidden layers:', num_hidden_layers)
    print('Neurons per layer:', neurons_per_layer)
    print('activations:', activations)

    # Build neural network (using LSTM for time series)
    model = tf.keras.Sequential()
    # Input layer
    model.add(tf.keras.layers.LSTM(neurons_per_layer[0], return_sequences=True, input_shape=(1, INPUT_SIZE)))

    # Hidden layers
    for i in range(1, num_hidden_layers-1):
        model.add(tf.keras.layers.LSTM(neurons_per_layer[i], return_sequences=True))
    # Output layer
    model.add(tf.keras.layers.LSTM(neurons_per_layer[num_hidden_layers-1]))
    model.add(tf.keras.layers.Dense(OUTPUT_SIZE))

    # Compile model
    opt = keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='mean_squared_error', optimizer=opt)

    # Reshape for LSTM
    Xtrain = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    ytrain = y_train.reshape(y_train.shape[0], 1, y_train.shape[1])
    Xtest = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    ytest = y_test.reshape(y_test.shape[0], 1, y_test.shape[1])

    # Train model
    model.fit(Xtrain, ytrain, epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, verbose=VERBOSE)

    # Save neural network
    structure_text = f"nuLayer_{num_hidden_layers+1}-neurons_{neurons_per_layer}-activation_{activations}.keras"
    model.save(structure_text)
    
    # Evaluate model fitness
    mse = model.evaluate(Xtest, ytest, verbose=VERBOSE)
    return -mse  

# Define variable boundaries (including number of hidden layers)
varbound = np.array([[MIN_HIDDEN_LAYERS, MAX_HIDDEN_LAYERS]] +
                    [[MIN_NEURONS_PER_LAYER, MAX_NEURONS_PER_LAYER]] * MAX_HIDDEN_LAYERS +
                    [[0, len(activation_list) - 1]] * MAX_HIDDEN_LAYERS)

# Define algorithm parameters
algorithm_param = {
    'max_num_iteration': max_num_iteration,
    'population_size': population_size,
    'mutation_probability': mutation_probability,
    'elit_ratio': elit_ratio,
    'crossover_probability': crossover_probability,
    'parents_portion': parents_portion,
    'crossover_type': crossover_type,
    'max_iteration_without_improv': None
}

# Create genetic algorithm optimizer
model = ga(function=fitness_function, dimension=1 + 2 * MAX_HIDDEN_LAYERS,
           variable_type='int', variable_boundaries=varbound, algorithm_parameters=algorithm_param)

# Run the optimization
model.run()

# Print best solution found
best_solution = model.result['variable']
print("Best solution:", best_solution)
print("Fitness:", -model.result['score'])  # Un-negate MSE for readability

# Extract and print neural network structure (considering dynamically determined layers)
num_hidden_layers = int(best_solution[0])
neurons_per_layer = [int(best_solution[i]) for i in range(1, num_hidden_layers + 1)]
activations = [activation_list[int(best_solution[MAX_HIDDEN_LAYERS + i])] for i in range(1, num_hidden_layers + 1)]
print("Neural Network Structure:", best_solution)
print("Number of Hidden Layers:", num_hidden_layers)
print("Neurons per Layer:", neurons_per_layer)
print("Activation Functions:", activations)
