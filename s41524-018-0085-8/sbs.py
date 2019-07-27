from sklearn.base import clone
from itertools import combinations
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        log_f = open("$HOME/log.txt", "wt") ##
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]

        count1 = 1 ##
        

        while dim > self.k_features:
            scores = []
            subsets = []

            count2 = 1 ##

            log_f.write("Dimension : %d\n\n" %dim) ##

            for p in combinations(self.indices_, r=dim -1):
                log_f.write("%-4d - %4d\n" %(count1, count2)) ##
                comb_set = [X.columns[i] for i in p] ##
                log_f.write(comb_set) ##
                log_f.write("\n") ##
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                log_f.write(score) ##
                log_f.write("\n\n") ##
                scores.append(score)
                subsets.append(p)
                count2 += 1 ##

            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)

            dim -= 1
            self.scores_.append(scores[best])
            best_comb_set = [X.columns[i] for i in subsets[best]] ##
            log_f.write("Best Combination : {}\nScore : {}\n\n\n".format(best_comb_set, scores[best])) ##
            count1 += 1 ##

        self.k_score_ = self.scores_[-1]
        log_f.close()

        return self

    def transform(self, X):
        return X[:, self.indicies_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score
        