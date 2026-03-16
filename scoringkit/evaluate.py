import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
import os
import json

def evaluate():
    submit_dir = '/app/input/res/' 
    reference_dir = '/app/input/ref/'
    output_dir = '/app/output/'

    submission = pd.read_csv(os.path.join(submit_dir, 'predictions.csv'))
    reference = pd.read_csv(os.path.join(reference_dir, 'test_reference.csv'))

    y_true = reference['label']
    y_pred = submission['label']
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='macro')

    scores = {
        'accuracy': acc,
        'f1': f1
    }

    with open(os.path.join(output_dir, 'scores.json'), 'w') as f:
        json.dump(scores, f)

if __name__ == "__main__":
    evaluate()