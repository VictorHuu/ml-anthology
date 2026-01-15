from json.encoder import INFINITY
import matplotlib.pyplot as plt
import numpy as np
import util
import os

from p05b_lwr import LocallyWeightedLinearRegression


def main(tau_values, train_path, valid_path, test_path, pred_path):
    """Problem 5(b): Tune the bandwidth paramater tau for LWR.

    Args:
        tau_values: List of tau values to try.
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    # Load training set
    x_train, y_train_target = util.load_dataset(train_path, add_intercept=True)

    # *** START CODE HERE ***
    # Search tau_values for the best tau (lowest MSE on the validation set)
    min_mse = float('inf')
    best_tau = 0
    x_eval, y_eval_target = util.load_dataset(valid_path, add_intercept=True)
    x_test, y_test_target = util.load_dataset(test_path, add_intercept=True)
    for tau in tau_values:
        clf = LocallyWeightedLinearRegression(tau)
        clf.fit(x_train, y_train_target)
        y_eval = clf.predict(x_eval)
        mse = np.mean((y_eval-y_eval_target)**2)
        print(f'MSE of validation set with tau being {tau}(05-b):',mse)
        plt.figure()
        plt.title(f'Locally weighted regression with tau being {tau} (05-b)')
        plt.scatter(x_eval[:,1],y_eval,c='red',marker='o')
        plt.scatter(x_eval[:,1],y_eval_target,c='grey',marker='o')
        plt.show()
        filename = os.path.basename(valid_path)
        name_without_ext = os.path.splitext(filename)[0]
        plt.savefig(f'image_{name_without_ext}_lwr_{tau}.png')
        plt.clf()
        plt.close()
        if mse< min_mse:
            min_mse=mse
            best_tau=tau
    # Fit a LWR model with the best tau value
    bclf = LocallyWeightedLinearRegression(best_tau)
    bclf.fit(x_train, y_train_target)
    # Run on the test set to get the MSE value
    y_test = clf.predict(x_test)
    mse = np.mean((y_test-y_test_target)**2)
    print(f'MSE of test set with tau being {best_tau}(05-b):',mse)
    # Save predictions to pred_path
    # Plot data
    
    # *** END CODE HERE ***
