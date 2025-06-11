#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May 20 15:12:29 2025

@author: ugurburak
"""

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.metrics import median_absolute_error, r2_score

veriseti = pd.read_csv('/Users/ugurburak/Desktop/Ders Notları/Machine Learning/veri setleri/Ann.csv')

X = veriseti.iloc[:,:-1].values
y= veriseti.iloc[:,-1].values

X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=0.3,random_state=0)

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = scaler.fit_transform(y_train.reshape(-1,1))
y_test = scaler.transform(y_test.reshape(-1,1))

# %%

from keras.models import Sequential
from keras.layers import Dense

model  = Sequential()
#ilk gizli katman
model.add(Dense(12, kernel_initializer='uniform',activation='relu',input_dim=5))

model.add(Dense(1, kernel_initializer='uniform',activation= None))

model.compile(optimizer='adam',loss='mean_squared_error',metrics=['mean_absolute_error'])

history = model.fit(X_train,y_train,validation_split=0.2,batch_size=10,epochs=10)
model.summary()

# %%  Test seti sonçlarının tahmini (Genel)

y_pred = model.predict(X_test)
print('R_2: ', r2_score(y_test, y_pred)) # burdan sonu var finalde

# %%

for i in model.layers:
    ilk_gizli_katman = model.layers[0].get_weights()
    cikti_katman = model.layers[1].get_weights()

olceklendirme = scaler.inverse_transform(y_pred.reshape(-1,1))

# %%

# grafik çizdirme ve metriklerini yaz.

plt.figure(figsize=(12,4))

plt.subplot(1, 2,1)
plt.plot(history.history['mean_absolute_error'],label ='Eğitim Doğruluğu')
plt.plot(history.history['val_mean_absolute_error'],label ='Test Doğruluğu')
plt.title('Doğruluk Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'],label ='Eğitim Kaybı')
plt.plot(history.history['val_loss'],label ='Test Kaybı')
plt.title('Kayıp Eğrisi')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend()
 

plt.show()








