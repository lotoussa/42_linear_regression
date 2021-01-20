import sys
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    def __init__(self):
        self.args = self._load_args()
        self.columns = ["x", "y"]
        self.x = []
        self.y = []
        self._x = []
        self._y = []
        self.m = 0.0
        self.b = 0.0
        self._m = 0.0
        self._b = 0.0
        self.error = []

    @staticmethod
    def _load_args():
        parser = argparse.ArgumentParser(usage="python3 %(prog)s [options] (-h:--help)",
                                         description="Proceed linear regression on dataframe.")
        parser.add_argument("-p", "--path", help="csv dataset to load", type=str, default="data.csv")
        parser.add_argument("-e", "--epochs", help="number of iterations", type=int, default=3000)
        parser.add_argument("-lr", "--learning-rate", help="learning rate", type=float, default=0.01)
        parser.add_argument("-r", "--random", help="size of random dataframe", type=int, default=-1)
        parser.add_argument("-v", "--plot", help="plot visualisation", action="store_true")
        parser.add_argument("-c", "--cost", help="cost plot visualisation", action="store_true")
        return parser.parse_args()

    @staticmethod
    def _load_csv(path):
        try:
            if path.split('.')[-1] != "csv":
                sys.exit(f"Dataset must be a .csv")
            df = pd.read_csv(path, na_filter=False)
            columns = df.columns.to_list()
            if len(columns) != 2:
                sys.exit(f"Expected 2 columns, have {len(columns)}.")
            x = df[columns[0]].to_numpy()
            y = df[columns[1]].to_numpy()
            return x, y, columns
        except:
            sys.exit(f"Error while parsing {path}")

    @staticmethod
    def _load_random(random):
        x = np.linspace(-1, 1, random) + np.random.normal(0, 0.25, random)
        y = np.linspace(-1, 1, random) + np.random.normal(0, 0.25, random)
        return x, y

    def _check_data(self):
        if self.args.random < 1 and self.args.random != -1:
            self.args.random = 2
        if self.args.epochs < 0:
            self.args.epochs = 100
        if self.args.learning_rate < 0.00001:
            self.args.learning_rate = 0.001
        elif self.args.learning_rate > 1:
            self.args.learning_rate = 1

    def load_data(self):
        self._check_data()
        if self.args.random != -1:
            self.x, self.y = self._load_random(self.args.random)
        else:
            self.x, self.y, self.columns = self._load_csv(self.args.path)

    @staticmethod
    def _m_grad(m, b, x, y):
        return sum(-2 * r * (y[idx] - (m * r + b)) for idx, r in enumerate(x)) / float(len(x))

    @staticmethod
    def _b_grad(m, b, x, y):
        return sum(-2 * (y[idx] - (m * r + b)) for idx, r in enumerate(x)) / float(len(x))

    @staticmethod
    def _error(m, b, x, y):
        return sum(((m * r + b) - y[idx]) ** 2 for idx, r in enumerate(x)) / float(len(x))

    @staticmethod
    def _standardize_data(data):
        imax = max(data)
        return [x / imax for x in data]

    def _readjust_data(self):
        self._m = self.b * max(self.y)
        self._b = (self.m * max(self.y)) / max(self.x)
        self.m = self._b
        self.b = self._m

    def gradient_descent(self):
        try:
            self._x = self._standardize_data(self.x)
            self._y = self._standardize_data(self.y)
        except:
            sys.exit("Error while processing data.")
        for _ in range(self.args.epochs):
            self.m = self.m - self.args.learning_rate * self._m_grad(self.m, self.b, self._x, self._y)
            self.b = self.b - self.args.learning_rate * self._b_grad(self.m, self.b, self._x, self._y)
            self.error.append(self.args.learning_rate * self._error(self.m, self.b, self._x, self._y))
        self._readjust_data()

    def save(self):
        try:
            with open("model.p", "wb") as f:
                pickle.dump((float(self._m), float(self._b)), f)
        except:
            sys.exit("Error while saving model.p")

    def _cost_plot(self):
        plt.xlabel("Iterations")
        plt.ylabel("global cost of error")
        plt.plot(range(self.args.epochs), self.error, '.r')
        plt.show()

    def _plot(self):
        plt.title("Linear Regression")
        plt.xlabel(self.columns[0])
        plt.ylabel(self.columns[1])
        line_x = [min(self.x), max(self.x)]
        line_y = [(self.m * i) + self.b for i in line_x]
        plt.plot(self.x, self.y, 'ob')
        plt.plot(line_x, line_y, 'chartreuse')
        plt.show()

    def show(self):
        print(f"theta0: {self._m}\n"
              f"theta1: {self._b}")
        if self.args.plot is True:
            self._plot()
        if self.args.cost is True:
            self._cost_plot()


if __name__ == "__main__":
    trainer = LinearRegression()
    trainer.load_data()
    trainer.gradient_descent()
    trainer.save()
    trainer.show()
