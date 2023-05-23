import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import mean_squared_error

model = load_model('model1.h5')
df = pd.read_csv('train.csv')

test_user_data = df['userId']
test_movie_data = df['movieId']

# Test model
y_pred = model.predict([test_user_data, test_movie_data])
y_true = df['rating'].values

#  Compute RMSE
rmse = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
print('\n\nTesting Result With Keras Deep Learning: {:.4f} RMSE'.format(rmse))