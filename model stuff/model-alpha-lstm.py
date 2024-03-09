import pandas as PD
# import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import matplotlib.dates as mdates
from keras.layers import LSTM, Dense
from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
# from sklearn import linear_model
from keras.models import Sequential
from keras.utils import plot_model
# import keras.backend as K
# from keras.callbacks import EarlyStopping
# from keras.optimizers import Adam
import yfinance as yf




df = yf.Ticker("^GSPC")
df = df.history(period = "max")

df.plot.line(y="Close", use_index=True)
plt.show()

#Set Target Variable
output_var = PD.DataFrame(df['Close'])
#Selecting the Features
features = ['Open', 'High', 'Low', 'Volume']

scaler = MinMaxScaler()
feature_transform = scaler.fit_transform(df[features])
feature_transform= PD.DataFrame(columns=features, data=feature_transform, index=df.index)
feature_transform.head()

timesplit= TimeSeriesSplit(n_splits=10)
for train_index, test_index in timesplit.split(feature_transform):
        X_train, X_test = feature_transform[:len(train_index)], feature_transform[len(train_index): (len(train_index)+len(test_index))]
        y_train, y_test = output_var[:len(train_index)].values.ravel(), output_var[len(train_index): (len(train_index)+len(test_index))].values.ravel()
        
#Process the data for LSTM
trainX =np.array(X_train)
testX =np.array(X_test)
X_train = trainX.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = testX.reshape(X_test.shape[0], 1, X_test.shape[1])

#Building the LSTM Model
lstm = Sequential()
lstm.add(LSTM(32, input_shape=(1, trainX.shape[1]), activation= 'relu', return_sequences=False))
lstm.add(Dense(1))
lstm.compile(loss= 'mean_squared_error', optimizer= 'adam')
plot_model(lstm, show_shapes=True, show_layer_names=True)

#Model Training
history=lstm.fit(X_train, y_train, epochs=100, batch_size=8, verbose=1, shuffle=False)

#LSTM Prediction
y_pred= lstm.predict(X_test)

#Predicted vs True Adj Close Value – LSTM
plt.plot(y_test, label='True Value')
plt.plot(y_pred, label='LSTM Value')
plt.title('Prediction by LSTM')
plt.xlabel('Time Scale')
plt.ylabel('Scaled USD')
plt.legend()
plt.show()

# Save the model
lstm.save("your_model.h5")

