import pandas as pd
from sklearn.dummy import DummyClassifier

class BaselineModel:
    def __init__(self):
        self.model = DummyClassifier(strategy='most_frequent', random_state=42)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):        
        return self.model.predict(X_test)

def main():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test_input.csv')

    X_train = train_df['sentence']
    y_train = train_df['label']
    X_test = test_df['sentence']

    classifier = BaselineModel()
    
    classifier.fit(X_train, y_train)

    predictions = classifier.predict(X_test)

    submission_df = test_df[['sentence']].copy()
    submission_df['label'] = predictions
    
    submission_df.to_csv('predictions.csv', index=False)

if __name__ == "__main__":
    main()