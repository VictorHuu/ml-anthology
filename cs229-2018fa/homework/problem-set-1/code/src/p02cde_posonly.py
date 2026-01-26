from ctypes.util import test
import numpy as np
from matplotlib import pyplot as plt
import util

import os
import sys

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'

def save_pred(y_pred, folder="results", filename="ds_pred.csv"):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    np.savetxt(path, y_pred, delimiter=",", fmt="%d")
    print(f"Predictions saved to {path}")

def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    # *** START CODE HERE ***
    x_train, t_train = util.load_dataset(train_path,label_col='t', add_intercept=True)
    _, y_train = util.load_dataset(train_path,label_col='y', add_intercept=True)
    x_test, t_test_target = util.load_dataset(test_path, label_col='t',add_intercept=True)
    _, y_test_target = util.load_dataset(test_path, label_col='y',add_intercept=True)
    x_eval, t_eval = util.load_dataset(valid_path, label_col='t',add_intercept=True)
    _, y_eval = util.load_dataset(valid_path, label_col='y',add_intercept=True)
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    theta = np.zeros(x_train.shape[1])
    classifier = LogisticRegression(theta_0=theta)
    classifier.fit(x_train,t_train)

    t_pred = classifier.predict(x_test)
    acc = np.sum((t_pred>0.5).astype(int)==t_test_target)/t_pred.size
    print("Accuracy of Part (c):",acc)
    save_pred(t_pred,pred_path)
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    
    theta = np.zeros(x_train.shape[1])
    classifier_d = LogisticRegression(theta_0=theta)
    classifier_d.fit(x_train,y_train)

    y_test_d = classifier_d.predict(x_test)
    acc = np.sum((y_test_d>0.5).astype(int)==y_test_target)/y_test_d.size
    print("Accuracy of Part (d):",acc)
    save_pred(y_test_d,pred_path)
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    
    y_eval_e = classifier_d.predict(x_eval)
    alpha = np.mean(y_eval_e[y_eval==1])
    print("α (e):",alpha)
    acc = np.sum((y_test_d/alpha>0.5).astype(int)==t_test_target)/y_test_d.size
    print("Accuracy of Part after rescale(e):",acc)
    # *** BEGIN PLOT ***
    plt.figure()
    x1 = np.linspace(x_test[:,1].min()-0.3*abs(x_test[:,1].min()), x_test[:,1].max()+0.3*abs(x_test[:,1].min()), 1000)
    x2 = np.linspace(x_test[:,2].min()-0.3*abs(x_test[:,2].min()), x_test[:,2].max()+0.3*abs(x_test[:,2].min()), 1000)
    X1, X2 = np.meshgrid(x1, x2)
    Z = classifier.predict(np.c_[np.ones(X1.ravel().shape),X1.ravel(), X2.ravel()]).reshape(X1.shape)
    Z2 = classifier_d.predict(np.c_[np.ones(X1.ravel().shape),X1.ravel(), X2.ravel()]).reshape(X1.shape)
    colors = np.where(y_test_target==1, 'red', 'blue')
    filename = os.path.basename(train_path)
    name_without_ext = os.path.splitext(filename)[0]
    #plt.style.use('seaborn')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    axes[0].set_title('Decision w/ Incomplete,positive-only labels(c)')
    axes[0].scatter(x_test[:,1], x_test[:,2], c=colors, s=60, marker='*')
    axes[0].contour(X1, X2, Z, levels=[0.5], colors='red', linewidths=3, linestyles='-')

    axes[1].set_title('Decision w/ Incomplete,positive-only labels(d)')
    axes[1].scatter(x_test[:,1], x_test[:,2], c=colors, s=60, marker='*')
    axes[1].contour(X1, X2, Z2, levels=[0.5], colors='red', linewidths=2, linestyles='-')

    axes[2].set_title('Decision w/ Incomplete,positive-only labels(e)')
    axes[2].scatter(x_test[:,1], x_test[:,2], c=colors, s=60, marker='*')
    axes[2].contour(X1, X2, Z2 / alpha, levels=[0.5], colors='red', linewidths=2, linestyles='-')

    plt.tight_layout()
    plt.savefig(f'image_{name_without_ext}_positive_only_all.png',dpi=80,bbox_inches='tight')
    plt.show()
    plt.clf()
    plt.close()
    # *** END PLOT ***
    # *** END CODER HERE

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python p02cde_posonly.py train.csv eval.csv pred.csv")
        sys.exit(1)
    train_path = sys.argv[1]
    eval_path = sys.argv[2]
    test_path = sys.argv[3]
    pred_path = sys.argv[4]
    main(train_path, eval_path, test_path,pred_path)