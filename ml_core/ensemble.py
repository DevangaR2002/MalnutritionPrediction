import numpy as np

class SoftVotingEnsemble:
    def __init__(self, rf, xgb, cat, threshold=0.1, weights=(0.4, 0.3, 0.3)):
        self.rf = rf
        self.xgb = xgb
        self.cat = cat
        self.threshold = threshold
        self.weights = np.array(weights)

    def predict_proba(self, X):
        rf_p = self.rf.predict_proba(X)[:, 1]
        xgb_p = self.xgb.predict_proba(X)[:, 1]
        cat_p = self.cat.predict_proba(X)[:, 1]

        weighted = (
            self.weights[0] * rf_p +
            self.weights[1] * xgb_p +
            self.weights[2] * cat_p
        )
        return np.vstack([1 - weighted, weighted]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= self.threshold).astype(int)
