import pandas as pd
import random
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

data = pd.read_csv("C:/Users/anuraj/Downloads/SaYoPillow.csv")

X = data.iloc[:,:-1].values
y = data['sl']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=27)
KNN_model = KNeighborsClassifier(n_neighbors=7)
KNN_model.fit(X_train, y_train)
KNN_prediction = KNN_model.predict(X_test)
print(X_test)

#print(KNN_prediction)
print("Accuracy : ",accuracy_score(KNN_prediction, y_test)*100)

filename = 'finalized-model.sav'
joblib.dump(KNN_model,filename)