import numpy as np
from matplotlib import pyplot as plt
import util

import sys
import os

from linear_model import LinearModel

def save_pred(y_pred, folder="results", filename="ds_pred.csv"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    np.savetxt(path, y_pred, delimiter=",", fmt="%d")
    print(f"Predictions saved to {path}")

def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    theta = np.zeros(x_train.shape[1])
    classifier = LogisticRegression(theta_0=theta)
    classifier.fit(x_train,y_train)

    x_eval, y_eval = util.load_dataset(eval_path, add_intercept=True)
    y_pred = classifier.predict(x_eval)
    print("y_pred",y_pred)
    acc = np.sum((y_pred>0.5).astype(int)==y_eval)/y_eval.size
    print("Accuracy:",acc)
    save_pred(y_pred,pred_path)
    # *** END CODE HERE ***
    # *** BEGIN PLOT ***
    #plt.style.use('seaborn')
    plt.figure()
    plt.title('Decision Boundary found by Logistic Regression')
    colors = np.where(y_train==1, 'red', 'blue')
    plt.scatter(x_train[:,1],x_train[:,2],c=colors,s=60,marker='*')
    # column 0 is 1 as intercept
    x1 = np.linspace(x_train[:,1].min(), x_train[:,1].max(), 1000)
    x2 = np.linspace(x_train[:,2].min(), x_train[:,2].max(), 1000)
    X1, X2 = np.meshgrid(x1, x2)
    Z = classifier.predict(np.c_[np.ones(X1.ravel().shape),X1.ravel(), X2.ravel()]).reshape(X1.shape)

    plt.contour(X1, X2, Z, levels=[0.5], colors='gold', linewidths=2,linestyles='-')
    plt.show()
    filename = os.path.basename(train_path)
    name_without_ext = os.path.splitext(filename)[0]
    plt.savefig(f'image_{name_without_ext}_discriminitive.png')
    plt.clf()
    plt.close()
    # *** END PLOT ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """
    @staticmethod
    def g(z):
        return 1/(1+np.exp(-z))

    def h(self,x):
        return self.g(x @ self.theta )

    def hessian(self,j,k,X):
        res = 0
        m = X.shape[0]
        for i in range(0,m):
            res += self.h(X[i])*(1-self.h(X[i]))*X[i][j]*X[i][k]
        return res/m

    def pd(self,j,X,y):
        res = 0
        m = X.shape[0]
        for i in range(m):
            res+=(y[i]-self.h(X[i]))*X[i][j]
        return -res/m
    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        m,n= x.shape

        for k in range(self.max_iter):
            H = np.zeros((n,n))
            v = np.zeros(m)
            for i in range(m):
                v[i]=self.h(x[i])*(1-self.h(x[i]))/m
            D = np.diag(v)
            H = x.T @ D @ x
            
            h_vals = self.h(x)
            PD = -(x.T @ (y - h_vals)) / m
            delta = np.linalg.solve(H,PD)

            old_theta = self.theta
            self.theta =self.theta - delta
            diff_norm1=np.linalg.norm(self.theta-old_theta,1)
            print(" theta @iter",k,":",self.theta)
            if diff_norm1< self.eps:
                break
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        print("final iter:",self.theta)
        return self.g(x @ self.theta)
        # *** END CODE HERE ***

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python p01b_logreg.py train.csv eval.csv pred.csv")
        sys.exit(1)
    train_path = sys.argv[1]
    eval_path = sys.argv[2]
    pred_path = sys.argv[3]
    main(train_path, eval_path, pred_path)
