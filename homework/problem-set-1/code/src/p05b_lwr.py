import matplotlib.pyplot as plt
import numpy as np
import util
import math

import sys
import os

from linear_model import LinearModel

def save_pred(y_pred, folder="results", filename="ds_pred.csv"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    np.savetxt(path, y_pred, delimiter=",", fmt="%d")
    print(f"Predictions saved to {path}")

def main(tau, train_path, eval_path):
    """Problem 5(b): Locally weighted regression (LWR)

    Args:
        tau: Bandwidth parameter for LWR.
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
    """
    # Load training set
    x_train, y_train_target = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Fit a LWR model
    clf = LocallyWeightedLinearRegression(tau)
    clf.fit(x_train, y_train_target)
    # Get MSE value on the validation set
    x_eval, y_eval_target = util.load_dataset(eval_path, add_intercept=True)
    y_eval = clf.predict(x_eval)
    y_train = clf.predict(x_train)
    mse = np.mean((y_eval-y_eval_target)**2)
    print("MSE of validation set (05-b):",mse)
    # Plot validation predictions on top of training set
    plt.figure()
    plt.title('Locally weighted regression(05-b)')
    plt.scatter(x_eval[:,1],y_eval,c='red',marker='o')
    plt.scatter(x_train[:,1],y_train,c='blue',marker='*')
    plt.scatter(x_eval[:,1],y_eval_target,c='grey',marker='o')
    plt.scatter(x_train[:,1],y_train_target,c='grey',marker='*')
    plt.show()
    filename = os.path.basename(eval_path)
    name_without_ext = os.path.splitext(filename)[0]
    plt.savefig(f'image_{name_without_ext}_lwr.png')
    plt.clf()
    plt.close()
    # No need to save predictions
    # Plot data
    # *** END CODE HERE ***


class LocallyWeightedLinearRegression(LinearModel):
    """Locally Weighted Regression (LWR).

    Example usage:
        > clf = LocallyWeightedLinearRegression(tau)
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def __init__(self, tau):
        super(LocallyWeightedLinearRegression, self).__init__()
        self.tau = tau
        self.x = None
        self.y = None

    def fit(self, x, y):
        """Fit LWR by saving the training set.

        """
        # *** START CODE HERE ***
        self.x=x
        self.y=y
        # *** END CODE HERE ***

    def predict(self, x):
        """Make predictions given inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m, n = x.shape
        m_train = self.x.shape[0]
        y= np.zeros(m)
        # Solve the eq to get the least square solution
        for i in range(m):
            weight= np.zeros(m_train)
            for j in range(m_train):
                diff = self.x[j]-x[i]
                weight[j]=math.exp(-np.sum(diff**2)/(2*self.tau**2))
            W = np.diag(weight)
            WX = W @ self.x
            theta = np.linalg.solve(self.x.T @ WX,WX.T @ self.y)
            y[i]=x[i] @ theta
        return y
        # *** END CODE HERE ***

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python p05b_lwr.py tau train.csv eval.csv ")
        sys.exit(1)

    tau = float(sys.argv[1])
    train_path = sys.argv[2]
    eval_path = sys.argv[3]

    main(tau, train_path, eval_path)
