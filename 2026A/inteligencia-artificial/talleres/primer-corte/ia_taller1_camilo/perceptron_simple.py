import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


class Perceptron:
    def __init__(self, learning_rate=0.1, epochs=100):
        self.lr = learning_rate
        self.epochs = epochs
        self.weights = None
        self.bias = None
        self.errors = []

    def activation(self, x):
        return np.where(x >= 0, 1, 0)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.epochs):
            errors = 0

            for i in range(n_samples):
                linear = np.dot(X[i], self.weights) + self.bias
                y_pred = self.activation(linear)
                update = self.lr * (y[i] - y_pred)
                self.weights += update * X[i]
                self.bias += update

                if update != 0:
                    errors += 1

            self.errors.append(errors)

            if errors == 0:
                break

    def predict(self, X):
        linear = np.dot(X, self.weights) + self.bias
        return self.activation(linear)


def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)


def plot_boundary(X, y, model, output_path):
    plt.figure(figsize=(7, 5))

    for label in np.unique(y):
        plt.scatter(X[y == label][:, 0], X[y == label][:, 1], label=f"Clase {label}")

    x_values = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)

    if model.weights[1] != 0:
        y_values = -(model.weights[0] * x_values + model.bias) / model.weights[1]
        plt.plot(x_values, y_values, linestyle="--", label="Frontera")

    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Perceptrón simple")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_errors(errors, output_path, title):
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(errors) + 1), errors, marker="o")
    plt.xlabel("Época")
    plt.ylabel("Errores")
    plt.title(title)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    results_dir = Path(__file__).resolve().parent / "resultados"
    results_dir.mkdir(exist_ok=True)

    X1 = np.array([
        [1, 2],
        [2, 3],
        [3, 3],
        [2, 1],
        [6, 5],
        [7, 8],
        [8, 6],
        [9, 7],
    ])
    y1 = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    X2 = np.array([
        [1, 7],
        [2, 8],
        [3, 8],
        [2, 6],
        [7, 2],
        [8, 3],
        [9, 2],
        [8, 1],
    ])
    y2 = np.array([0, 0, 0, 0, 1, 1, 1, 1])

    model1 = Perceptron()
    model1.fit(X1, y1)
    pred1 = model1.predict(X1)
    print("Dataset 1")
    print("Predicciones:", pred1)
    print("Accuracy:", accuracy(y1, pred1))
    plot_boundary(X1, y1, model1, results_dir / "perceptron_dataset1.png")
    plot_errors(model1.errors, results_dir / "perceptron_error_dataset1.png", "Errores por época - Dataset 1")

    model2 = Perceptron()
    model2.fit(X2, y2)
    pred2 = model2.predict(X2)
    print("Dataset 2")
    print("Predicciones:", pred2)
    print("Accuracy:", accuracy(y2, pred2))
    plot_boundary(X2, y2, model2, results_dir / "perceptron_dataset2.png")
    plot_errors(model2.errors, results_dir / "perceptron_error_dataset2.png", "Errores por época - Dataset 2")
