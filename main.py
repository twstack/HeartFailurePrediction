import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.compose import ColumnTransformer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer
from sklearn.metrics import classification_report

data = pd.read_csv('heart_failure.csv')
print(data.info())
print(Counter(data["DEATH_EVENT"]))

y = data["DEATH_EVENT"]
x = data[['age','anaemia','creatinine_phosphokinase','diabetes','ejection_fraction','high_blood_pressure','platelets','serum_creatinine','serum_sodium','sex','smoking','time']]

X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)

ct = ColumnTransformer([("numeric", StandardScaler(), ['age','creatinine_phosphokinase','ejection_fraction','platelets','serum_creatinine','serum_sodium','time'])])

X_train = ct.fit_transform(X_train)
X_test = ct.transform(X_test)

model = Sequential()
model.add(InputLayer(input_shape=(X_train.shape[1],)))
model.add(Dense(8, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model.fit(X_train, Y_train, epochs = 100, batch_size = 16, verbose = 1)
loss, acc = model.evaluate(X_test, Y_test, verbose=0)

y_estimate = model.predict(X_test)
y_estimate = [1 if prob >= 0.5 else 0 for prob in y_estimate]

print(classification_report(Y_test, y_estimate))
