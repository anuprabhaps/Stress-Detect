import pandas as pd
import seaborn as sns
import joblib
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import roc_auc_score

data = pd.read_csv("C:/Users/anuraj/Downloads/SaYoPillow.csv")

print(data.isnull().sum())



print(data.describe().T)


sns.countplot(data = data, x = "sl")
plt.xlabel('Stress level')
plt.ylabel('Count')

plt.figure(figsize=(10,5))
sns.lineplot(x='rr',y='sl',data=data,hue='sl',palette='deep')
plt.title('Snoring rate vs Stress level')
plt.ylabel('Snoring rate')
plt.xlabel('Stress level')
plt.show()

plt.figure(figsize=(10,5))
sns.lineplot(x='rr',y='sl',data=data,hue='sl',palette='deep')
plt.title('Respiration Rate vs Stress Level')
plt.xlabel("Respiration Rate")
plt.ylabel('Stress Level')
plt.show()

plt.figure(figsize=(10,5))
sns.lineplot(x='t',y='sl',data=data,hue='sl',palette='deep')
plt.xlabel("Body Temperature")
plt.ylabel('Stress Level')
plt.title('Body Temperature vs Stress Level')
plt.show()

plt.figure(figsize=(10,5))
sns.lineplot(x='bo',y='sl',data=data,hue='sl',palette='deep')
plt.xlabel("Blood Oxygen")
plt.ylabel('Stress Level')
plt.title('Blood Oxygen vs Stress Level')
plt.show()

plt.figure(figsize=(10,5))
sns.lineplot(x='rem',y='sl',data=data,hue='sl',palette='deep')
plt.xlabel("Eye Movement")
plt.ylabel('Stress Level')
plt.title('Eye Movement vs Stress Level')
plt.show()

plt.figure(figsize=(10,5))
sns.lineplot(x='sr.1',y='sl',data=data,hue='sl',palette='deep')
plt.xlabel("Sleeping Hours")
plt.ylabel('Stress Level')
plt.title('Sleeping Hours vs Stress Level')
plt.show()

plt.figure(figsize=(10,5))
sns.lineplot(x='hr',y='sl',data=data,hue='sl',palette='deep')
plt.xlabel("Heart Rate")
plt.ylabel('Stress Level')
plt.title('Heart rate vs Stress Level')
plt.show()

plt.figure(figsize=(10,10))
tc = data.corr()
sns.heatmap(tc,annot=True)
plt.title('Heatmap')
plt.show()

X = data.iloc[:,:-1].values
y = data['sl']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=27)

mms = MinMaxScaler()
xtrain_s = mms.fit_transform(X_train)
xtest_s = mms.fit_transform(X_test)



knn = KNeighborsClassifier()
svc = SVC()
rfc = RandomForestClassifier()
dtc = DecisionTreeClassifier()
lr = LogisticRegression()

knn.fit(X_train,y_train)
svc.fit(X_train,y_train)
dtc.fit(X_train,y_train)
rfc.fit(X_train,y_train)
lr.fit(X_train,y_train)

lrp = lr.predict(X_test)
knnp = knn.predict(X_test)
rfcp = rfc.predict(X_test)
dtcp = dtc.predict(X_test)
svcp = svc.predict(X_test)

predicts = [lrp,svcp,knnp,rfcp,dtcp]

for i in predicts:
    a = accuracy_score(i,y_test)
    print("Accuracy score :",a*100)
for j in predicts:   
    b = balanced_accuracy_score(j,y_test)
    print("Balanced accuracy score :",b*100)    

filename = 'finalized-model.sav'
joblib.dump(knn,filename)