
from matplotlib.pyplot import axis
import numpy as np

# Sigmoid activation function, squashes input 'z' to a range between 0 and 1
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

# Converts integer labels (e.g., 0, 1, 2) to one-hot encoded format
# This creates a binary matrix representation for labels
def int_to_onehot(y, num_labels):
    # Initialize a zero matrix of shape (num_samples, num_labels)
    one_hot = np.zeros((y.shape[0], num_labels))

    # Set the appropriate index to 1 for each label
    for i, val in enumerate(y):
        one_hot[i, val] = 1

    return one_hot

# Define a simple Neural Network with a single hidden layer
class NeuralNetMLP():

    def __init__(self, num_features, num_hidden, num_classes, random_seed=123):
        super().__init__()

        self.num_classes = num_classes
        self.num_hidden = num_hidden

        # Initialize the random number generator with a seed for reproducibility
        rng = np.random.RandomState(random_seed)

        # Initialize weights and biases for the hidden layer
        # Weights are normally distributed with mean=0 and std=0.1
        # Biases are initialized to 0
        self.weight_hidden = rng.normal(
            loc=0.0, scale=0.1, size=(num_hidden, num_features)
        )
        self.bias_hidden = np.zeros(num_hidden)

        # Initialize weights and biases for the output layer
        self.weight_output = rng.normal(
            loc=0.0, scale=0.1, size=(num_classes, num_hidden))
        self.bias_output = np.zeros(num_classes)

    # Forward pass of the network
    def forward(self, x):
        # Hidden layer calculations
        # input 'x' shape: [num_samples, num_features]
        # weights shape: [num_hidden, num_features].T
        # output shape: [num_samples, num_hidden]

        # Compute linear combination for the hidden layer
        z_hidden = np.dot(x, self.weight_hidden.T) + self.bias_hidden
        # Apply sigmoid activation function
        a_hidden = sigmoid(z_hidden)

        # Output layer calculations
        # a_hidden shape: [num_samples, num_hidden]
        # weights shape: [num_classes, num_hidden].T
        # output shape: [num_samples, num_classes]
        z_output = np.dot(a_hidden, self.weight_output.T) + self.bias_output
        # Apply sigmoid activation function
        a_output = sigmoid(z_output)

        # Return hidden and output activations
        return a_hidden, a_output

    # Backward pass (backpropagation) to compute gradients
    def backward(self, x, a_hidden, a_output, y):
        # Compute gradients for the output and hidden layers

        # Convert labels 'y' to one-hot encoded format
        y_onehot = int_to_onehot(y, self.num_classes)

        # Compute derivative of the loss with respect to the output activation
        # Mean Squared Error Loss: dLoss/dOutput = 2 * (output - target) / num_samples
        d_loss__d_a_output = 2.0 * (a_output - y_onehot) / y.shape[0]

        # Derivative of the sigmoid function for the output layer
        d_a_output__d_z_output = a_output * (1.0 - a_output)

        # Compute the error term for the output layer (delta)
        delta_output = d_loss__d_a_output * d_a_output__d_z_output

        # Gradient for the output weights
        # a_hidden shape: [num_samples, num_hidden]
        # delta_output.T shape: [num_classes, num_samples]
        # d_loss__d_weight_output shape: [num_classes, num_hidden]
        d_z_output__d_weight_output = a_hidden
        d_loss__d_weight_output = np.dot(delta_output.T, d_z_output__d_weight_output)
        # Gradient for the output biases
        d_loss__d_bias_output = np.sum(delta_output, axis=0)

        # Backpropagate the error to the hidden layer
        # Gradient from the output to the hidden layer
        d_z_output__d_a_hidden = self.weight_output

        # Compute the error term for the hidden layer
        d_loss__d_a_hidden = np.dot(delta_output, d_z_output__d_a_hidden)

        # Derivative of the sigmoid function for the hidden layer
        d_a_hidden__d_z_hidden = a_hidden * (1.0 - a_hidden)

        # Calculate gradient for the hidden layer weights
        d_z_hidden__d_weight_hidden = x
        d_loss__d_weight_hidden = np.dot((d_loss__d_a_hidden * d_a_hidden__d_z_hidden).T, d_z_hidden__d_weight_hidden)

        # Gradient for the hidden biases
        d_loss__d_bias_hidden = np.sum(d_loss__d_a_hidden * d_a_hidden__d_z_hidden, axis=0)

        # Return the gradients for weight and bias updates
        return (d_loss__d_weight_output, d_loss__d_bias_output,
                d_loss__d_weight_hidden, d_loss__d_bias_hidden)
