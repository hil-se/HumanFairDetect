import numpy as np
from scipy.sparse import csr_matrix

# [1] Lowy, Andrew, Rakesh Pavan, Sina Baharlouei, Meisam Razaviyayn, and Ahmad Beirami. "FERMI: Fair Empirical Risk Minimization via Exponential R\'enyi Mutual Information." arXiv preprint arXiv:2102.12586 (2021).
# Reproduced with code at: https://github.com/optimization-for-data-driven-science/FERMI

class FERMI:
    def __init__(self, lam = 30000, num_iterations = 500, step_size = 0.0001, stopping = 0.001):
        self.lam = lam
        self.num_iterations = num_iterations
        self.step_size = step_size
        self.stopping = stopping

    def grad_sigmoid(self, x):
        return np.exp(-x) / ((1.0 + np.exp(-x)) * (1.0 + np.exp(-x)))

    def sigmoid(self, x):  # P(Y = 1 | X, \theta) Input: X\theta
        return 1.0 / (1.0 + np.exp(-x))

    def fit(self, X, y, S, sample_weight=None):
        if type(X)==csr_matrix:
            X = X.toarray()
        n, d = X.shape
        self.theta = np.zeros((d, 1))
        if sample_weight is None:
            sample_weight = np.array([1]*len(y))

        # Computing the gradient of regularizer

        for iter_num in range(self.num_iterations):
            logits = np.dot(X, self.theta)
            probs = self.sigmoid(logits)
            grad_probs = self.grad_sigmoid(logits)

            g1 = np.dot(X.T, ((probs.flatten() - y) * sample_weight).reshape(n,1))

            P_Y1 = sum(np.dot(probs.T, sample_weight)) / sum(sample_weight)
            P_Y0 = 1 - P_Y1

            grad_Y1 = np.dot(grad_probs.flatten() * sample_weight, X).reshape(d, 1)
            grad_Y1 /= sum(sample_weight)
            grad_Y0 = - grad_Y1

            regularizer_grad = np.zeros(self.theta.shape)

            for j in np.unique(S):
                indicator_function = (S == j) * 1

                indicator_function = indicator_function * sample_weight
                total = sum(sample_weight)
                # total = n

                number_of_s = sum(indicator_function)
                P_S = number_of_s / total

                P_Y1S = np.dot(indicator_function.T, probs)[0] / number_of_s
                P_Y0S = 1 - P_Y1S

                q_1j = P_Y1S * np.sqrt(P_S) / np.sqrt(P_Y1)
                q_0j = P_Y0S * np.sqrt(P_S) / np.sqrt(P_Y0)

                # Computing the gradient with respect to self.theta

                conditional_grad_probs = np.multiply(indicator_function, grad_probs.flatten())


                grad_Y1S = np.dot(conditional_grad_probs.flatten(), X).reshape(d, 1)
                grad_Y1S /= number_of_s
                grad_Y0S = - grad_Y1S

                # Gradient of q_ij with respect to self.theta:
                grad_q1j = np.sqrt(P_S) / P_Y1 * (np.sqrt(P_Y1) * grad_Y1S - P_Y1S / (2 * np.sqrt(P_Y1)) * grad_Y1)
                grad_q0j = np.sqrt(P_S) / P_Y0 * (np.sqrt(P_Y0) * grad_Y0S - P_Y0S / (2 * np.sqrt(P_Y0)) * grad_Y0)

                regularizer_grad += 2 * q_1j * grad_q1j + 2 * q_0j * grad_q0j

            total_grad = g1 + self.lam * regularizer_grad
            self.theta -= self.step_size * total_grad
            if np.linalg.norm(total_grad, 2) < self.stopping:
                break


    def predict_proba(self, X):
        if type(X)==csr_matrix:
            X = X.toarray()
        logits = np.dot(X, self.theta)
        probs = self.sigmoid(logits)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        preds = np.array([1 if p>=0.5 else 0 for p in probs])
        return preds



