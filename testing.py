import time
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import mean_squared_error

model1 = load_model('model1.h5')
model2 = load_model('model2.h5')
df = pd.read_csv('test.csv')

test_user_data = df['userId']
test_movie_data = df['movieId']

# Test 1 model
start = time.time()
y_pred = model1.predict([test_user_data, test_movie_data])
result_time_1 = time.time() - start
y_true = df['rating'].values

#  Compute RMSE
rmse_1 = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
print('\n\nTesting Result With Keras Deep Learning: {:.4f} RMSE'.format(rmse_1))


# Test 2 model
start = time.time()
y_pred = model2.predict([test_user_data, test_movie_data])
result_time_2 = time.time() - start
y_true = df['rating'].values

#  Compute RMSE
rmse_2 = np.sqrt(mean_squared_error(y_pred=y_pred, y_true=y_true))
print('\n\nTesting Result With Keras Deep Learning: {:.4f} RMSE'.format(rmse_2))

with open('results.txt', 'w') as f:
    print('Keras Deep Learning: rmse=', "%.1f" % rmse_1, ', duration=', "%.1f" % result_time_1, file=f)
    print('Keras Matrix-Factorization: rmse=', "%.1f" % rmse_2, ', duration=', "%.1f" % result_time_2, file=f)
