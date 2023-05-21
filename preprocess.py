import pandas as pd
import numpy as np
from datetime import date

movie_metadata = pd.read_csv('movies_metadata.csv', low_memory=False)[['id','original_title', 'overview', 'vote_count']].dropna()
movie_metadata.set_index('id', inplace=True)

df = pd.read_csv('ratings_small.csv')
df['timestamp'] = [date.fromtimestamp(x) for x in df['timestamp']]
df.rename(columns={"timestamp": "date"}, inplace=True)

df_filtered = df.drop('date', axis=1).sample(frac=1).reset_index(drop=True)
df_filtered.to_csv('data.csv')
