import numpy as np
import matplotlib.pyplot as plt

X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])

np.random.seed(42)
W1 = np.random.uniform(-1, 1, (2, 2))
b1 = np.zeros((1, 2))
W2 = np.random.uniform(-1, 1, (2, 1))
b2 = np.zeros((1, 1))

def sigmoid(x): 
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x): 
    return x * (1 - x)

epochs = 5000
learning_rate = 0.5
losses = []

for epoch in range(epochs):
    hidden_layer_input = np.dot(X, W1) + b1
    hidden_layer_output = sigmoid(hidden_layer_input)
    
    output_layer_input = np.dot(hidden_layer_output, W2) + b2
    predicted_output = sigmoid(output_layer_input)
    
    error = y - predicted_output
    loss = np.mean(0.5 * error**2)
    losses.append(loss)
    
    d_output = error * sigmoid_derivative(predicted_output)
    d_hidden = d_output.dot(W2.T) * sigmoid_derivative(hidden_layer_output)
    
    W2 += hidden_layer_output.T.dot(d_output) * learning_rate
    b2 += np.sum(d_output, axis=0, keepdims=True) * learning_rate
    W1 += X.T.dot(d_hidden) * learning_rate
    b1 += np.sum(d_hidden, axis=0, keepdims=True) * learning_rate

print("Final Predictions:\n", np.round(predicted_output, 3))

plt.figure(figsize=(7, 5), dpi=300)
plt.plot(range(epochs), losses)
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (Loss)')
plt.title('Training Loss for XOR using 1 Hidden Layer')
plt.grid(True)
plt.tight_layout()
plt.show()