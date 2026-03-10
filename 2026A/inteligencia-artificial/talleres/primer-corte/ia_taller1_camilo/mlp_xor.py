import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

np.random.seed(42)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    return x * (1 - x)


class MLP:
    def __init__(self, input_size, hidden_layers, output_size, learning_rate=0.5):
        self.layers = [input_size] + hidden_layers + [output_size]
        self.lr = learning_rate
        self.weights = []
        self.biases = []

        for i in range(len(self.layers) - 1):
            w = np.random.uniform(-1, 1, (self.layers[i], self.layers[i + 1]))
            b = np.random.uniform(-1, 1, (1, self.layers[i + 1]))
            self.weights.append(w)
            self.biases.append(b)

        self.loss = []

    def forward(self, X):
        activations = [X]
        a = X

        for w, b in zip(self.weights, self.biases):
            z = np.dot(a, w) + b
            a = sigmoid(z)
            activations.append(a)

        return activations

    def backward(self, X, y, activations):
        output = activations[-1]
        error = y - output
        deltas = [error * sigmoid_derivative(output)]

        for i in range(len(self.weights) - 1, 0, -1):
            delta = np.dot(deltas[0], self.weights[i].T) * sigmoid_derivative(activations[i])
            deltas.insert(0, delta)

        for i in range(len(self.weights)):
            self.weights[i] += self.lr * np.dot(activations[i].T, deltas[i])
            self.biases[i] += self.lr * np.sum(deltas[i], axis=0, keepdims=True)

    def train(self, X, y, epochs=5000):
        for _ in range(epochs):
            activations = self.forward(X)
            loss = np.mean((y - activations[-1]) ** 2)
            self.loss.append(loss)
            self.backward(X, y, activations)

    def predict(self, X):
        output = self.forward(X)[-1]
        return (output >= 0.5).astype(int), output


def plot_loss(loss_values, output_path):
    plt.figure(figsize=(7, 5))
    plt.plot(loss_values)
    plt.xlabel("Época")
    plt.ylabel("Loss")
    plt.title("Entrenamiento MLP XOR")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parent / "resultados"
    results_dir.mkdir(exist_ok=True)

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1],
    ])

    y = np.array([
        [0],
        [1],
        [1],
        [0],
    ])

    model = MLP(2, [4], 1, learning_rate=0.5)
    model.train(X, y, epochs=5000)
    pred, raw = model.predict(X)

    print("Predicciones binarias:")
    print(pred)
    print("Salidas continuas:")
    print(raw)

    plot_loss(model.loss, results_dir / "mlp_loss.png")
