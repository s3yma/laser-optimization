# -*- coding: utf-8 -*-
"""
Created on Fri May  3 00:40:12 2024

@author: 90546
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Veri setinizi yükleyin
veri = pd.read_csv(r'C:\Users\90546\Desktop\Dosyalarım\Veri Bilimi Çalışma\Machine_Learning_Laser\sample_data.csv')

# Özellikler ve etiketleri ayırın
X = veri[['K', 'alphap', 'meanE', 'varE', 'skewE', 'kurtE']]
y = veri['skewE']

# Eğitim ve test setlerini oluşturun
X_egitim, X_test, y_egitim, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Regressor modelini oluşturun ve eğitin
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_egitim, y_egitim)

# Test seti üzerinde tahmin yapın
tahminler = rf_regressor.predict(X_test)

# Modelin performansını değerlendirin
mse = mean_squared_error(y_test, tahminler)
print("Ortalama Kare Hata (MSE):", mse)