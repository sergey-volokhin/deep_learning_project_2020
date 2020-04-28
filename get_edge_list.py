import pandas as pd

reviews = pd.read_csv('reviews.tsv', delimiter='\t')
df = reviews[['critic_id', 'movie_id', 'score']]
print(df)
df = df[df['score']!=3.0]
print(df)
df['edge_type'] = df['score'].apply(lambda x: 2 if x > 3 else 1)
df = df[['edge_type', 'critic_id', 'movie_id']]

result1 = [f'{id} 1' for id in df['critic_id'].unique()]
result2 = [f'{id} 2' for id in df['movie_id'].unique()]

open('node_type.txt', 'w').write('\n'.join(result1 + result2))
