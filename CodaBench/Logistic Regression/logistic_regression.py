import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

def main():
    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test_input.csv')

    X_train = train_df['sentence']
    y_train = train_df['label']
    X_test = test_df['sentence']

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced'))
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    submission_df = test_df.copy()
    submission_df['label'] = predictions
    
    submission_df = submission_df[['sentence', 'label']]
    submission_df.to_csv('predictions.csv', index=False)
    

if __name__ == "__main__":
    main()