import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

class BaselineModel:

    def __init__(self):
        self.model = DummyClassifier(strategy='stratified', random_state=42)

    def fit(self, x_train, y_train):
        self.model.fit(x_train, y_train)

    def predict(self, X_test):        
        return self.model.predict(X_test)