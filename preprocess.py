import pandas as pd
from datetime import date

df = pd.read_csv('ratings_small.csv')
df['timestamp'] = [date.fromtimestamp(x) for x in df['timestamp']]
df.rename(columns={"timestamp": "date"}, inplace=True)

df_filtered = df.drop('date', axis=1).sample(frac=1).reset_index(drop=True)

user_id_mapping = {id:i for i, id in enumerate(df_filtered['userId'].unique())}
movie_id_mapping = {id:i for i, id in enumerate(df_filtered['movieId'].unique())}

# Testingsize
n = int(len(df_filtered) * 0.1)

# Split train- & testset
df_train = df_filtered[:-n]
df_test = df_filtered[-n:]

df_train.to_csv('train.csv', index=False)
df_test.to_csv('test.csv', index=False)

users = max(user_id_mapping)
movies = max(movie_id_mapping)

with open('index.txt', 'w') as f:
    print(users, movies, file=f, end='')
