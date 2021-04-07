# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 14:56:22 2021

@author: gvill
"""

import pandas as pd
import numpy as np 
from matplotlib import pyplot as plt
import seaborn as sns
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense, LSTM, LeakyReLU, Dropout
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler

Produit=["BTC-EUR","ETH-EUR","XTZ-EUR"] 

tab_decision=[]


for i in range(len(Produit)):
    
    crypto = yf.Ticker(Produit[i])

    # get historical market data
    #“1d”, “5d”, “1mo”, “3mo”, “6mo”, “1y”, “2y”, “5y”, “10y”, “ytd”, “max”

    Periode="2y"
    data = crypto.history(period=Periode)
    data['Date']=data.index
    data=data.rename_axis('index').reset_index()
    del data['index']

    price = data[['Close']]

    plt.figure(figsize = (15,9))
    plt.plot(price)
    plt.xticks(range(0, data.shape[0],50), data['Date'].loc[::50],rotation=45)
    plt.title(Produit[i] + " Price",fontsize=18, fontweight='bold')
    plt.xlabel('Date',fontsize=18)
    plt.ylabel('Close Price (EUR)',fontsize=18)
    plt.show()
    
    price.info()
        
    min_max_scaler = MinMaxScaler()
        
    norm_data = min_max_scaler.fit_transform(price.values)
        
    """
    print(f'Real: {price.values[0]}, Normalized: {norm_data[0]}')
    print(f'Real: {price.values[500]}, Normalized: {norm_data[500]}')
    print(f'Real: {price.values[1200]}, Normalized: {norm_data[1200]}')
    """
    
    
    def univariate_data(dataset, start_index, end_index, history_size, target_size):
      data = []
      labels = []
    
      start_index = start_index + history_size
      if end_index is None:
        end_index = len(dataset) - target_size
    
      for i in range(start_index, end_index):
        indices = range(i-history_size, i)
        # Reshape data from (history_size,) to (history_size, 1)
        data.append(np.reshape(dataset[indices], (history_size, 1)))
        labels.append(dataset[i+target_size])
      return np.array(data), np.array(labels)
    
    past_history = 5
    future_target = 0
    
    TRAIN_SPLIT = int(len(norm_data) * 0.8)
    
    
    x_train, y_train = univariate_data(norm_data,
                                       0,
                                       TRAIN_SPLIT,
                                       past_history,
                                       future_target)
    
    x_test, y_test = univariate_data(norm_data,
                                     TRAIN_SPLIT,
                                     None,
                                     past_history,
                                     future_target)
    
    
    num_units = 100
    learning_rate = 0.0002
    activation_function = 'sigmoid'
    adam = Adam(lr=learning_rate)
    loss_function = 'mse'
    batch_size = 5
    num_epochs = 500
    # Initialize the RNN
    model = Sequential()
    model.add(LSTM(units = num_units, activation=activation_function, input_shape=(None, 1)))
    model.add(LeakyReLU(alpha=0.5))
    model.add(Dropout(0.1))
    model.add(Dense(units = 1))
    
    # Compiling the RNN
    model.compile(optimizer=adam, loss=loss_function)
    
    
    model.summary()
    
    
    # Using the training set to train the model
    history = model.fit(
        x_train,
        y_train,
        validation_split=0.1,
        batch_size=batch_size,
        epochs=num_epochs,
        shuffle=False
    )
    
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs = range(len(loss))
    
    plt.figure()
    
    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title("Training and Validation Loss")
    plt.legend()
    
    plt.show()
    
    
    
    
    original = pd.DataFrame(min_max_scaler.inverse_transform(y_test))
    predictions = pd.DataFrame(min_max_scaler.inverse_transform(model.predict(x_test)))
    
    ax = sns.lineplot(x=original.index, y=original[0], label="Test Data", color='royalblue')
    ax = sns.lineplot(x=predictions.index, y=predictions[0], label="Prediction", color='tomato')
    ax.set_title(Produit[i] +  ' price', size = 14, fontweight='bold')
    ax.set_xlabel("Days", size = 14)
    ax.set_ylabel("Cost (EUR)", size = 14)
    ax.set_xticklabels('', size=10)
    
    def decision():
        if predictions[0][len(predictions)-1]>predictions[0][len(predictions)-2]:
            tab_decision.append(["Buy ", Produit[i]])
        else:
            
            tab_decision.append(["Don't Buy ",Produit[i]])
    decision()

print(tab_decision)