from re import X
import matplotlib.pyplot as plt
import numpy as np
import os

PLOT_COLORS = ['red', 'green', 'blue', 'orange']  # Colors for your plots
K = 4           # Number of Gaussians in the mixture model
NUM_TRIALS = 3  # Number of trials to run (can be adjusted for debugging)
UNLABELED = -1  # Cluster label for unlabeled data points (do not change)


def main(is_semi_supervised, trial_num):
    """Problem 3: EM for Gaussian Mixture Models (unsupervised and semi-supervised)"""
    print('Running {} EM algorithm...'
          .format('semi-supervised' if is_semi_supervised else 'unsupervised'))

    # Load dataset
    train_path = os.path.join('..', 'data', 'ds3_train.csv')
    x, z = load_gmm_dataset(train_path)
    x_tilde = None

    if is_semi_supervised:
        # Split into labeled and unlabeled examples
        labeled_idxs = (z != UNLABELED).squeeze()
        x_tilde = x[labeled_idxs, :]   # Labeled examples
        z = z[labeled_idxs, :]         # Corresponding labels
        x = x[~labeled_idxs, :]        # Unlabeled examples

    # *** START CODE HERE ***
    # (1) Initialize mu and sigma by splitting the m data points uniformly at random
    # into K groups, then calculating the sample mean and covariance for each group
    m = len(x)
    x = np.random.permutation(x)
    groups = np.array_split(x, K)

    mu = np.array([np.mean(g, axis=0) for g in groups])
    sigma = np.array([np.cov(g, rowvar=False) for g in groups])
    
    # (2) Initialize phi to place equal probability on each Gaussian
    # phi should be a numpy array of shape (K,)
    phi = np.full(K, 1/K)
    # (3) Initialize the w values to place equal probability on each Gaussian
    # w should be a numpy array of shape (m, K)
    w=  np.full((m,K), 1/K)
    # *** END CODE HERE ***

    if is_semi_supervised:
        w = run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma)
    else:
        w = run_em(x, w, phi, mu, sigma)

    # Plot your predictions
    z_pred = np.zeros(m)
    if w is not None:  # Just a placeholder for the starter code
        for i in range(m):
            z_pred[i] = np.argmax(w[i])

    plot_gmm_preds(x, z_pred, is_semi_supervised, plot_id=trial_num)


def run_em(x, w, phi, mu, sigma):
    """Problem 3(d): EM Algorithm (unsupervised).

    See inline comments for instructions.

    Args:
        x: Design matrix of shape (m, n).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    eps = 1e-3  # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    m, n = x.shape
    K = w.shape[1]

    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE
        # (1) E-step: Update your estimates in w
        for j in range(K):
            w[:, j] = phi[j] * multivariate_gaussian(x, mu[j], sigma[j])

        w /= w.sum(axis=1, keepdims=True)

        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi = np.mean(w, axis=0)
        for j in range(K):
            mu[j] = np.sum(w[:, j][:, np.newaxis] * x, axis=0) / np.sum(w[:, j])
            diff = x - mu[j]
            sigma[j] = (w[:, j][:, np.newaxis] * diff).T @ diff / np.sum(w[:, j])

        # (3) Compute the log-likelihood of the data to check for convergence.
        # By log-likelihood, we mean `ll = sum_x[log(sum_z[p(x|z) * p(z)])]`.
        # We define convergence by the first iteration where abs(ll - prev_ll) < eps.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        likelihoods = np.zeros(m)
        for i in range(m):
            temp = 0
            for j in range(K):
                temp += phi[j] * multivariate_gaussian(x[i:i+1], mu[j], sigma[j])[0]
            likelihoods[i] = temp


        ll = np.sum(np.log(likelihoods))
        # *** END CODE HERE
        it += 1
        print(it)
    return w


def run_semi_supervised_em(x, x_tilde, z, w, phi, mu, sigma):
    """Problem 3(e): Semi-Supervised EM Algorithm.

    See inline comments for instructions.

    Args:
        x: Design matrix of unlabeled examples of shape (m, n).
        x_tilde: Design matrix of labeled examples of shape (m_tilde, n).
        z: Array of labels of shape (m_tilde, 1).
        w: Initial weight matrix of shape (m, k).
        phi: Initial mixture prior, of shape (k,).
        mu: Initial cluster means, list of k arrays of shape (n,).
        sigma: Initial cluster covariances, list of k arrays of shape (n, n).

    Returns:
        Updated weight matrix of shape (m, k) resulting from semi-supervised EM algorithm.
        More specifically, w[i, j] should contain the probability of
        example x^(i) belonging to the j-th Gaussian in the mixture.
    """
    # No need to change any of these parameters
    alpha = 20.  # Weight for the labeled examples
    eps = 1e-3   # Convergence threshold
    max_iter = 1000

    # Stop when the absolute change in log-likelihood is < eps
    # See below for explanation of the convergence criterion
    it = 0
    ll = prev_ll = None
    m, n = x.shape
    m_tilde, n_tilde= x_tilde.shape
    K = w.shape[1]
    
    while it < max_iter and (prev_ll is None or np.abs(ll - prev_ll) >= eps):
        pass  # Just a placeholder for the starter code
        # *** START CODE HERE ***
        # (1) E-step: Update your estimates in w
        w_tilde = np.zeros((m_tilde,K))
        for j in range(K):
            w[:, j] = phi[j] * multivariate_gaussian(x, mu[j], sigma[j])
        for j in range(K):
            w_tilde[:, j] = (z.flatten()==j)# Indicator
        w /= w.sum(axis=1, keepdims=True)
        # (2) M-step: Update the model parameters phi, mu, and sigma
        phi = (np.sum(w, axis=0) + alpha*np.sum(w_tilde,axis=0)) / (m + alpha * m_tilde)
        
        for j in range(K):
            mu_unlabeled = np.sum(w[:, j][:, np.newaxis] * x, axis=0)
            mu_labeled = np.sum(w_tilde[:, j][:, np.newaxis] * x_tilde, axis=0)
            mu[j] = (mu_unlabeled + alpha * mu_labeled) / (np.sum(w[:, j]) + alpha * np.sum(w_tilde[:, j]))
            
            diff_unlabeled = x - mu[j]
            sigma_unlabeled = (w[:, j][:, np.newaxis] * diff_unlabeled).T @ diff_unlabeled

            diff_labeled = x_tilde - mu[j]
            sigma_labeled =  (w_tilde[:, j][:, np.newaxis] * diff_labeled).T @ diff_labeled
            sigma[j] = (sigma_unlabeled + alpha*sigma_labeled) / (np.sum(w[:, j]) + alpha*np.sum(w_tilde[:, j]))
        # (3) Compute the log-likelihood of the data to check for convergence.
        # Hint: Make sure to include alpha in your calculation of ll.
        # Hint: For debugging, recall part (a). We showed that ll should be monotonically increasing.
        prev_ll = ll
        likelihoods = np.zeros(m)
        for i in range(m):
            temp = 0
            for j in range(K):
                temp += phi[j] * multivariate_gaussian(x[i:i+1], mu[j], sigma[j])[0]
            likelihoods[i] = temp

        # labeled contribution to log-likelihood
        labeled_ll = 0
        for j in range(K):
            mask = (z.flatten() == j)
            if np.sum(mask) > 0:
                labeled_ll += np.sum(np.log(multivariate_gaussian(x_tilde[mask], mu[j], sigma[j])))

        ll = np.sum(np.log(likelihoods)) + alpha * labeled_ll
        # *** END CODE HERE ***
        it+=1
        print(it)
    return w


# *** START CODE HERE ***
# Helper functions
def multivariate_gaussian(x,mu,sigma):
        n= x.shape[1]
        diff = x-mu
        det_sigma=np.linalg.det(sigma)
        inv_sigma = np.linalg.inv(sigma)
        norm_const=1/np.sqrt((2*np.pi)**n*det_sigma)
        exponent = -0.5 * np.sum(diff @ inv_sigma * diff,axis=1)
        return norm_const * np.exp(exponent)
# *** END CODE HERE ***


def plot_gmm_preds(x, z, with_supervision, plot_id):
    """Plot GMM predictions on a 2D dataset `x` with labels `z`.

    Write to the output directory, including `plot_id`
    in the name, and appending 'ss' if the GMM had supervision.

    NOTE: You do not need to edit this function.
    """
    plt.figure(figsize=(12, 8))
    plt.title('{} GMM Predictions'.format('Semi-supervised' if with_supervision else 'Unsupervised'))
    plt.xlabel('x_1')
    plt.ylabel('x_2')

    for x_1, x_2, z_ in zip(x[:, 0], x[:, 1], z):
        color = 'gray' if z_ < 0 else PLOT_COLORS[int(z_)]
        alpha = 0.25 if z_ < 0 else 0.75
        plt.scatter(x_1, x_2, marker='.', c=color, alpha=alpha)

    file_name = 'p03_pred{}_{}.pdf'.format('_ss' if with_supervision else '', plot_id)
    save_path = os.path.join('output', file_name)
    plt.savefig(save_path)


def load_gmm_dataset(csv_path):
    """Load dataset for Gaussian Mixture Model (problem 3).

    Args:
         csv_path: Path to CSV file containing dataset.

    Returns:
        x: NumPy array shape (m, n)
        z: NumPy array shape (m, 1)

    NOTE: You do not need to edit this function.
    """

    # Load headers
    with open(csv_path, 'r') as csv_fh:
        headers = csv_fh.readline().strip().split(',')

    # Load features and labels
    x_cols = [i for i in range(len(headers)) if headers[i].startswith('x')]
    z_cols = [i for i in range(len(headers)) if headers[i] == 'z']

    x = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=x_cols, dtype=float)
    z = np.loadtxt(csv_path, delimiter=',', skiprows=1, usecols=z_cols, dtype=float)

    if z.ndim == 1:
        z = np.expand_dims(z, axis=-1)

    return x, z


if __name__ == '__main__':
    np.random.seed(229)
    # Run NUM_TRIALS trials to see how different initializations
    # affect the final predictions with and without supervision
    for t in range(NUM_TRIALS):
        main(is_semi_supervised=False, trial_num=t)

        # *** START CODE HERE ***
        # Once you've implemented the semi-supervised version,
        # uncomment the following line.
        # You do not need to add any other lines in this code block.
        # main(is_semi_supervised=True, trial_num=t)
        # *** END CODE HERE ***
