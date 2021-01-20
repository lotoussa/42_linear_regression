import sys
import pickle


class Prediction:
    def __init__(self):
        self.mileage = 0.0
        self.theta0 = 0.0
        self.theta1 = 0.0
        self.price = 0.0

    def get_input(self):
        try:
            mileage = float(input("Predict the price of a car based on its mileage.\n"
                                  "    Provide mileage in km: "))
            if mileage < 0:
                raise ValueError
            self.mileage = mileage
        except ValueError:
            sys.exit("Wrong mileage format.")

    @staticmethod
    def _validate_model(model):
        if not isinstance(model, tuple) or len(model) != 2 \
                or not isinstance(model[0], float) or not isinstance(model[1], float):
            return False
        return True

    def load_model(self, path):
        try:
            with open(path, "rb") as f:
                model = pickle.load(f)
            if not self._validate_model(model):
                raise Exception
            self.theta0, self.theta1 = model
        except:
            sys.exit("Model not created or invalid.")

    def estimate(self):
        self.price = self.theta0 + (self.theta1 * self.mileage)
        print(f"Estimated price of the car with the following assumption:\n"
              f"    θ0 + (θ1 * mileage) = {self.price}")


if __name__ == "__main__":
    predict = Prediction()
    predict.get_input()
    predict.load_model("model.p")
    predict.estimate()
