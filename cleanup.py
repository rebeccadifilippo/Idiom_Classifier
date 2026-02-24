import pandas as pd

df = pd.read_csv('data/idioms_data.csv')


df['id'] = df['id'].fillna(method='ffill')
df['class'] = df['class'].fillna(method='ffill')

#group by sentence id and combine tokens into a full sentence
def merge_tokens(tokens):
    tokens = [str(t) for t in tokens if pd.notna(t)]
    sentence = ' '.join(tokens)
    sentence = sentence.replace(' ,', ',').replace(' .', '.').replace(' `` ', '"').replace(" '' ", '"')
    return sentence

sentences = df.groupby(['id', 'class'])['token'].apply(merge_tokens).reset_index()
sentences = sentences.rename(columns={'token': 'sentence', 'class': 'label'})

n_total = 1200
class_counts = sentences['label'].value_counts()
class_ratios = class_counts / class_counts.sum()
samples_per_class = (class_ratios * n_total).round().astype(int)

sampled = pd.concat([
    sentences[sentences['label'] == cls].sample(n=min(n, len(sentences[sentences['label']==cls])), random_state=42)
    for cls, n in samples_per_class.items()
])

sampled = sampled.sample(frac=1, random_state=42).reset_index(drop=True)

sampled.to_csv('data/idioms_sentences_random_stratified.csv', index=False, columns=['sentence', 'label'])

print("Done! Saved as data/idioms_sentences_random_stratified.csv")