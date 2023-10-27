from sklearn.linear_model import LogisticRegression


class LR:
    def __init__(self):
        self.clf = LogisticRegression(max_iter=1000, solver='newton-cholesky')
        self.params = {'model__C': [1e-4, 1e-3, 1e-2, 1e-1, 1]}