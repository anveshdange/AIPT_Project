
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader.data as data
import yfinance as yf
import streamlit as st 
import keras

from keras.models import load_model
from pathlib import Path
from typing import List
from typing import Any

TICKER: str = "AAPL"
START: str = "2009-12-31"
END: str = "2022-12-31"

st.title("Stock Trend Prediction") 
user_input = st.text_input("Enter Stock Ticker", TICKER) 

print("Downloading Data")
df: pd.DataFrame = pd.DataFrame(yf.download(user_input, start=START, end=END))
FILE: str = f"{user_input}.csv"
# os.makedirs("data", exist_ok=True)
FILE_PATH: Path = os.path.join("data", FILE)
print(f'Converting Data to CSV: {FILE}')

# df.to_csv(FILE_PATH)
print(f"{FILE} created Successfully")

# Describing data
st.subheader(f"Data from \"{START}\" to \"{END}\"")
st.subheader(f"Statistics of the \"{user_input}\" Stock Ticker")
st.write(df.describe()) 

# Visualizations
st.subheader("Closing Price vs Time Chart") 
ma100 = df.Close.rolling(100).mean() 
ma200 = df.Close.rolling(200).mean() 
fig = plt.figure(figsize=(12, 6))
plt.plot(df.Close, 'b', label="Stock Price")
plt.plot(ma100, 'r', label='100 SMA')
plt.plot(ma200, 'g', label='200 SMA')
plt.legend()
st.pyplot(fig)

# Splitting the data into training and testing sets
data_training: pd.DataFrame = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing: pd.DataFrame = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

# scaling the data
from sklearn.preprocessing import MinMaxScaler

scaler : MinMaxScaler = MinMaxScaler(feature_range=(0, 1))
data_training_array: np.ndarray = scaler.fit_transform(data_training)

# Splitting data into X_train and y_train 
X_train: List[Any] = list()
y_train: List[Any] = list()

i: int;
for i in range(100, data_training_array.shape[0]) :
    X_train.append(data_training_array[i-100:i])
    y_train.append(data_training_array[i, 0])

# loading the model 
model = load_model('LSTM_Model.h5')

# testing our model 
past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, pd.DataFrame(data_testing)], ignore_index=True)
input_data = scaler.fit_transform(final_df) 

X_test = []
y_test = []

for i in range(100, input_data.shape[0]) :
  X_test.append(input_data[i-100 : i])
  y_test.append(input_data[i, 0])

X_test, y_test = np.array(X_test), np.array(y_test)
y_predicted = model.predict(X_test)

scale_factor: float = 1/scaler.scale_[0]
y_predicted = y_predicted * scale_factor
y_test = y_test * scale_factor

FIGURES: Path = Path("Graphs")
# function to save graohs
def save_graph(path: Path) -> Path :
    os.makedirs(FIGURES, exist_ok = True)
    path: Path = os.path.join(FIGURES, path)
    plt.savefig(f"{path}")
    print(f"Image saved at: {path}")
    return path

# Plotting the Graph
plt.figure(figsize=(12, 6))
plt.plot(y_test, 'b', label='Original Price')
plt.plot(y_predicted, 'r', label="Predicted Price")
plt.xlabel("Time")
plt.ylabel(f"\"{user_input}\" Price")
plt.legend()
FILE: Path = Path(f"{user_input}_Predictions.png")
path: Path = save_graph(FILE)

# Final Graph 
st.subheader("Model Predictions") 
fig2 = plt.figure(figsize=(12,6)) 
plt.plot(y_test, 'b', label="Original Price") 
plt.plot(y_predicted, 'r', label="Predicted Price") 
plt.xlabel("Time")
plt.ylabel("f\"{user_input}\" Price")
plt.legend()
st.pyplot(fig2)