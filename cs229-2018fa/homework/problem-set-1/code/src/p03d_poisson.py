import numpy as np
import util
import math
import os
import sys
from linear_model import LinearModel

def save_pred(y_pred, folder="results", filename="ds_pred.csv"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    np.savetxt(path, y_pred, delimiter=",", fmt="%d")
    print(f"Predictions saved to {path}")

def main(lr, train_path, eval_path, pred_path):
    """Problem 3(d): Poisson regression with gradient ascent.

    Args:
        lr: Learning rate for gradient ascent.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    # Fit a Poisson Regression model
    # Run on the validation set, and use np.savetxt to save outputs to pred_path
    theta = np.zeros(x_train.shape[1])
    classifier = PoissonRegression(theta_1=theta, alpha=lr)
    classifier.fit(x_train, y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    y_pred = classifier.predict(x_eval)
    print("y_pred", y_pred)
    rmse = np.sqrt(np.mean((y_pred - y_eval)**2))
    cv_rmse = rmse/ np.mean(y_eval)
    print("CV(RMSE)(d):", cv_rmse)
    save_pred(y_pred, pred_path)
    # *** END CODE HERE ***

class PoissonRegression(LinearModel):
    """Poisson Regression.

    Example usage:
        > clf = PoissonRegression(step_size=lr)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, theta_1=None, step_size=0.2, max_iter=100, eps=1e-5,
                 theta_0=None, verbose=True, alpha=0.1):
        super().__init__(step_size, max_iter, eps, theta_0, verbose)
        self.theta = theta_1 if theta_1 is not None else np.zeros(1)
        self.alpha = alpha

    # canonical response function of the Poisson 
    def h(self, x):
        # Clip to prevent overflow
        z = np.clip(x @ self.theta, -50, 50)
        return np.exp(z)

    def fit(self, x, y):
        """Run gradient ascent to maximize likelihood for Poisson regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        # In case of divergence
        for k in range(self.max_iter):
            print(" theta @iter", k, ":", self.theta)
            old_theta = self.theta.copy()
            for i in range(m):
                self.theta = self.theta + (self.alpha / m) * x[i] * (y[i] - self.h(x[i]))
            
            diff_norm1 = np.linalg.norm(self.theta - old_theta, 1)
            if diff_norm1 < self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Floating-point prediction for each input, shape (m,).
        """
        # *** START CODE HERE ***
        print("final iter:", self.theta)
        return np.round(self.h(x)).astype(int)
        # *** END CODE HERE ***

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python p03d_poisson.py lr train.csv eval.csv pred/")
        sys.exit(1)

    lr = float(sys.argv[1])
    train_path = sys.argv[2]
    eval_path = sys.argv[3]
    pred_path = sys.argv[4]

    main(lr, train_path, eval_path, pred_path)