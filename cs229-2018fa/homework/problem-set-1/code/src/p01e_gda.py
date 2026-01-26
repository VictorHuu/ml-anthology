import numpy as np
from matplotlib import pyplot as plt
import util

import sys
import os

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)

    # *** START CODE HERE ***
    theta = np.zeros(x_train.shape[1])
    classifier = GDA(theta_0=theta,theta_1=theta)
    classifier.fit(x_train,y_train)
    # theta_0 is calculated manually so the intercept can't be added
    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=False)
    y_pred = classifier.predict(x_eval)
    print("y_pred",y_pred)
    acc = np.sum(y_pred==y_eval)/y_eval.size
    print("Accuracy:",acc)
    # *** END CODE HERE ***
    # *** BEGIN PLOT ***
    #plt.style.use('seaborn')
    plt.figure()
    colors = np.where(y_train==1, 'red', 'blue')
    plt.title('Decision Boundary found by GDA')
    plt.scatter(x_train[:,0],x_train[:,1],c=colors,s=60,marker='*')

    x1 = np.linspace(x_train[:,0].min(), x_train[:,0].max(), 1000)
    x2 = np.linspace(x_train[:,1].min(), x_train[:,1].max(), 1000)
    X1, X2 = np.meshgrid(x1, x2)
    Z = classifier.predict(np.c_[X1.ravel(), X2.ravel()]).reshape(X1.shape)

    plt.contour(X1, X2, Z, levels=[0.5], colors='black', linewidths=2,linestyles='--')
    plt.show()
    filename = os.path.basename(train_path)
    name_without_ext = os.path.splitext(filename)[0]
    plt.savefig(f'image_{name_without_ext}_generative.png')
    plt.clf()
    plt.close()
    # *** END PLOT ***

class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    def __init__(self, theta_1=None,step_size=0.2, max_iter=100, eps=1e-5,
                 theta_0=None, verbose=True):
        super().__init__(step_size, max_iter, eps,theta_0, verbose)
        self.theta_0=theta_1
        
    @staticmethod
    def g(z):
        return 1/(1+np.exp(-z))

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m,n= x.shape
        X0=x[y==0]
        X1=x[y==1]

        hat_phi=len(X1)/m
        hat_mu0=X0.mean(axis=0)
        hat_mu1=X1.mean(axis=0)
        hat_sigma=((X0-hat_mu0).T@(X0-hat_mu0)+(X1-hat_mu1).T @ (X1-hat_mu1))/m
        print("hat_phi:",hat_phi)
        print("hat_mu_0:",hat_mu0)
        print("hat_mu_1:",hat_mu1)
        print("hat_sigma:",hat_sigma)
        hat_sigma_inv = np.linalg.inv(hat_sigma)
        self.theta = hat_sigma_inv @ (hat_mu1-hat_mu0)
        self.theta_0= 0.5*(hat_mu0.T @ hat_sigma_inv @ hat_mu0-hat_mu1.T @ hat_sigma_inv @ hat_mu1)+np.log(hat_phi/(1-hat_phi))
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        return (self.g(x @ self.theta+self.theta_0.T)>0.5).astype(int)
        # *** END CODE HERE

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python p01b_logreg.py train.csv eval.csv pred.csv")
        sys.exit(1)
    train_path = sys.argv[1]
    eval_path = sys.argv[2]
    pred_path = sys.argv[3]
    main(train_path, eval_path, pred_path)

