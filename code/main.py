import os.path

import pandas as pd
import numpy as np
from keras import Sequential
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
# from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense
import time


# Mesurer le temps de début
start_time = time.time()

# Charger les données
# url = "https://archive.ics.uci.edu/ml/machine-learning-databases/abalone/abalone.data"
url = os.path.join('file/abalone.data')
column_names = ['Sex', 'Length', 'Diameter', 'Height', 'WholeWeight', 'ShuckedWeight', 'VisceraWeight', 'ShellWeight', 'Rings']
data = pd.read_csv(url, names=column_names)

# Remplacer les valeurs de sexe par des colonnes booléennes
data = pd.get_dummies(data, columns=['Sex'])

# Normaliser les caractéristiques
features = data.drop('Rings', axis=1)
labels = data['Rings'] / 30  # Normaliser l'âge entre 0 et 1

# Diviser les données en ensembles d'apprentissage et de test
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Normaliser les données
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Construire le modèle de réseau de neurones
model = Sequential()
model.add(Dense(4, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(1, activation='sigmoid'))

# Compiler le modèle
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.1), loss='mean_squared_error')

# Entraîner le modèle
model.fit(X_train, y_train, epochs=10000, batch_size=32, verbose=1)

# Évaluer le modèle
train_loss = model.evaluate(X_train, y_train, verbose=0)
test_loss = model.evaluate(X_test, y_test, verbose=0)

# Mesurer le temps de fin
end_time = time.time()
execution_time = end_time - start_time

# Afficher les résultats
print(f"Erreur quadratique moyenne sur l'ensemble d'apprentissage: {train_loss}")
print(f"Erreur quadratique moyenne sur l'ensemble de test: {test_loss}")
print(f"Durée d'exécution du code: {execution_time} secondes")

# Pour convertir la sortie normalisée en âge réel, multiplier par 30
predictions = model.predict(X_test) * 30
actual_ages = y_test * 30
