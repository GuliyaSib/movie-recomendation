import pandas as pd
import numpy as np
# To create deep learning models
from keras.layers import Input, Embedding, Reshape, Dot, Concatenate, Dense, Dropout
from keras.models import Model
from sklearn.metrics import mean_squared_error


df_filterd = pd.read_csv('train.csv')

# Testingsize
n = 10000

# Split train- & testset
df_train = df_filterd[:-n]
df_test = df_filterd[-n:]

# Split train- & testset
# Create user- & movie-id mapping
with open('index.txt') as f:
    users, movies = f.read().split()

users, movies = int(users), int(movies)

# Create correctly mapped train- & testset
train_user_data = df_train['userId']
train_movie_data = df_train['movieId']
test_user_data = df_test['userId']
test_movie_data = df_test['movieId']

# Deep Learning With Keras
embedding_size = 10

# Setup variables
user_embedding_size = 20
movie_embedding_size = 10


##### Create model
# Set input layers
user_id_input = Input(shape=[1], name='user')
movie_id_input = Input(shape=[1], name='movie')

# Create embedding layers for users and movies
user_embedding = Embedding(output_dim=user_embedding_size, 
                           input_dim=users + 1, 
                           input_length=1, 
                           name='user_embedding')(user_id_input)
movie_embedding = Embedding(output_dim=movie_embedding_size, 
                            input_dim=movies + 1,
                            input_length=1, 
                            name='item_embedding')(movie_id_input)

# Reshape the embedding layers
user_vector = Reshape([user_embedding_size])(user_embedding)
movie_vector = Reshape([movie_embedding_size])(movie_embedding)

# Concatenate the reshaped embedding layers
concat = Concatenate()([user_vector, movie_vector])

# Combine with dense layers
dense = Dense(256)(concat)
y = Dense(1)(dense)

# Setup model
model = Model(inputs=[user_id_input, movie_id_input], outputs=y)
model.compile(loss='mse', optimizer='adam')


# Fit model
model.fit([train_user_data, train_movie_data],
          df_train['rating'],
          batch_size=256, 
          epochs=1,
          validation_split=0.3,
          shuffle=True)

# Test model
y_pred = model.predict([test_user_data, test_movie_data])
y_true = df_test['rating'].values

model.save('model1.h5')


# Matrix Factorisation With Keras And Gradient Descent

# Get input variable-sizes
embedding_size = 20


##### Create model
# Set input layers
user_id_input = Input(shape=[1], name='user')
movie_id_input = Input(shape=[1], name='movie')

# Create embedding layers for users and movies
user_embedding = Embedding(output_dim=embedding_size, 
                           input_dim=users + 1,
                           input_length=1, 
                           name='user_embedding')(user_id_input)
movie_embedding = Embedding(output_dim=embedding_size, 
                            input_dim=movies + 1,
                            input_length=1, 
                            name='item_embedding')(movie_id_input)

# Reshape the embedding layers
user_vector = Reshape([embedding_size])(user_embedding)
movie_vector = Reshape([embedding_size])(movie_embedding)

# Compute dot-product of reshaped embedding layers as prediction
y = Dot(1, normalize=False)([user_vector, movie_vector])

# Setup model
model_2 = Model(inputs=[user_id_input, movie_id_input], outputs=y)
model_2.compile(loss='mse', optimizer='adam')


# Fit model
model_2.fit([train_user_data, train_movie_data],
          df_train['rating'],
          batch_size=256, 
          epochs=3,
          validation_split=0.1,
          shuffle=True)

# Test model
y_pred = model_2.predict([test_user_data, test_movie_data])
y_true = df_test['rating'].values

model_2.save('model2.h5')
