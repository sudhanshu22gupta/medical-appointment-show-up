from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from xgboost import XGBClassifier
from skopt import BayesSearchCV

import pandas as pd

class BinaryClassifier:

    def __init__(self):
        self.class_weight = None

    def fit_classifier(self, X_train, y_train):
        """
         Fit the classifier to the training data.
         
         @param X_train - The training data of shape [ n_samples n_features ]
         @param y_train - The target values of shape [ n_samples
        """
        self.classifier.fit(X_train, y_train)
    
    def set_class_weights(self):
        """
         Set the class_weights for the classifier inversely proportional to class frequencies in the input data.
        """
        self.class_weight = 'balanced'

    def predict_classifier(self, X):
        """
         Predict class labels for X. This is equivalent to calling `classifier.predict` method of the underlying classifier.
         
         @param X - pandas. DataFrame of shape [ n_samples n_features ]
         
         @return numpy. ndarray of shape [ n_samples ] or
        """
        return self.classifier.predict(X)

    def predict_proba_classifier(self, X):
        """
         Predict class probabilities for X. This is equivalent to calling `classifier. predict_proba` method of the underlying classifier.
         
         @param X - pandas. DataFrame of shape [ n_samples n_features ]
         
         @return numpy. ndarray of shape [ n_samples n_classes
        """
        return self.classifier.predict_proba(X)

    def predict_at_threshold(self, X, clf_threshold):
        """
         Predict class labels at a given threshold.
         
         @param X - pandas. DataFrame of shape [ n_samples n_features ]
         @param clf_threshold - threshold at which to classify each sample
         
         @return numpy. ndarray of shape [ n_samples ]
        """
        return (self.predict_proba_classifier(X)[:,1] > clf_threshold).astype(int)

    def evaluate_classifier(self, y_true, y_pred):
        """
        Evaluate the classifier and print the results.
        
        @param y_true - Array - like of shape = [ n_samples ] Ground truth ( correct ) target values.
        @param y_pred - Array - like of shape = [ n_samples ] Estimated targets as returned by a classifier
        
        @return Tuple of precision recall f1 and accuracy
        """

        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        # print(f"precision={precision}\nrecall={recall}\nf1={f1}\naccuracy={accuracy}")

        return precision, recall, f1, accuracy

    def predict_per_threshold(self, X, y_true, threshold_values, plot=True):
        """
         Predict per threshold and record metrics. Predict the classifiers for each threshold in threshold_values and record the precision recall f1 and accuracy.
         
         @param X - pandas. DataFrame of shape [ n_samples n_features ]
         @param y_true - labels of X as a vector of length n
         @param threshold_values - list of threshold values to predict for
         @param plot - whether to plot the results or not. Default is True
         
         @return pandas. DataFrame with columns : threshold precision
        """
        
        df_results = pd.DataFrame()
        # predict per threshold
        # predict_at_threshold X clf_threshold and evaluate the classifier at the threshold
        for clf_threshold in threshold_values:
            # get y_pred at threshold = clf_threshold
            y_pred = self.predict_at_threshold(X, clf_threshold)
            precision, recall, f1, accuracy = self.evaluate_classifier(y_true, y_pred)

            # record metrics
            df_results = pd.concat([
                    df_results, 
                    pd.DataFrame.from_dict({
                        "threshold": [clf_threshold], 
                        "precision": [precision], 
                        "recall": [recall],
                        "f1": [f1],
                        "accuracy": [accuracy],
                        })])
        df_results.reset_index(drop=True, inplace=True)
        
        # plot the results of the function
        if plot:
            df_results.plot(y=["precision", "recall", "f1", "accuracy"], x="threshold")

        return df_results

class LogisticRegressionClf(BinaryClassifier):
    def __init__(self):
        super().__init__()
        self.classifier = LogisticRegression(
            C=1, 
            penalty='l2', 
            max_iter=1000, 
            random_state=22,
            class_weight=self.class_weight,
        )

class RandomForestClf(BinaryClassifier):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = RandomForestClassifier(class_weight=self.class_weight)

    def hyperparmater_tuning(self, X_train, y_train, param_space, n_iter=20, cv=5, n_jobs=8):
        """
         Bayesian hyperparmater tuning method.
         
         @param X_train - training data shape [ n_samples n_features ]
         @param y_train - target values shape [ n_samples ]
         @param param_space - parameter space used to compute parameters.
         @param n_iter - number of iterations default 20. Use 20 for no limit
         @param cv - number of cross validation steps default 5. Use cv = 5 for NLT
         @param n_jobs
        """
        self.bayes_search = BayesSearchCV(
            self.classifier,
            param_space,
            n_iter=n_iter,
            cv=cv,
            n_jobs=n_jobs,
        )
        self.bayes_search.fit(X_train, y_train)
        self.classifier = RandomForestClassifier(**self.bayes_search.best_params_)
        self.fit_classifier(X_train, y_train)

class MLPClf(BinaryClassifier):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = MLPClassifier(
            hidden_layer_sizes = [512, 256, 64],
            warm_start=False, 
            max_iter=1000, 
            verbose=True,
            class_weight=self.class_weight,
        )

class XGBClf(BinaryClassifier):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = XGBClassifier()
