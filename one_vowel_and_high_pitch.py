import pandas as pd
import numpy as np

data = pd.read_csv("K:/mp3 music/one_vowel_and_high_pitch/final.csv")


data['label'].value_counts()


data=data.replace({"label":{"Healthy":0,"Unhealthy":1}})


data=data.drop(columns=['Unnamed: 0'])





data.sample(frac=1)

X=data.iloc[:,1:27]
Y=data.iloc[:,27:]




from sklearn.model_selection import train_test_split


x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.4,random_state=4)


import keras


from keras.models import Sequential




from keras.layers import Dense


classifier = Sequential()


classifier.add(Dense(64, activation = 'relu', input_dim = 26))
classifier.add(Dense(256, activation = 'relu'))
classifier.add(Dense(256, activation = 'relu'))
classifier.add(Dense(512, activation = 'relu'))
classifier.add(Dense(1024, activation = 'relu'))
classifier.add(Dense(2048, activation = 'relu'))
classifier.add(Dense(1, activation = 'sigmoid'))
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
classifier.summary()
classifier.fit(X, Y, epochs = 100,validation_split=0.3)
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)


from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: %f' % accuracy)
precision = precision_score(y_test, y_pred)
print('Precision: %f' % precision)

recall = recall_score(y_test, y_pred)
print('Recall: %f' % recall)

f1 = f1_score(y_test, y_pred)
print('F1 score: %f' % f1)


from sklearn.metrics import classification_report


print(classification_report(y_test, y_pred))
from imblearn.combine import SMOTEENN



sm = SMOTEENN()
X_resampled1, y_resampled1 = sm.fit_resample(X,Y)


xr_train1,xr_test1,yr_train1,yr_test1=train_test_split(X_resampled1, y_resampled1,test_size=0.2)



from sklearn.ensemble import RandomForestClassifier

model_rf_smote=RandomForestClassifier(n_estimators=100, criterion='gini', random_state = 100,max_depth=6, min_samples_leaf=8)




model_rf_smote.fit(xr_train1,yr_train1)






yr_predict1 = model_rf_smote.predict(xr_test1)




model_score_r1 = model_rf_smote.score(xr_test1, yr_test1)



print(model_score_r1)
print(classification_report(yr_test1, yr_predict1))
print(confusion_matrix(yr_test1, yr_predict1))



from xgboost import XGBClassifier



model = XGBClassifier()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
predictions = [round(value) for value in y_pred]

accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


model = XGBClassifier()
model.fit(xr_train1,yr_train1)



y_pred = model.predict(xr_test1)
predictions = [round(value) for value in y_pred]


print(classification_report(yr_test1, y_pred))
print(confusion_matrix(yr_test1, y_pred))




