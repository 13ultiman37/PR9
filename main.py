import numpy as np
import pandas as pd
import sklearn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, GridSearchCV
import matplotlib.pyplot as plt
import matplotlib
import plotly.express as px

matplotlib.use('TkAgg')
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

data = load_digits(as_frame=True)

predictors = data.data
target = data.target
target_names = data.target_names

print(predictors.head(5), '\n\nЦелевая переменная')
print(target.head(5))
print('Искомая цифра:\n', target_names)

x_train, x_test, y_train, y_test = train_test_split(
    predictors, target, train_size=0.8, shuffle=True, random_state=271)

print('Размер признаков обучающей выборки: ', x_train.shape, '\n',
      'Размер для признаков тестовой выборки: ', x_test.shape, '\n',
      'Размер для целевого показателя обучающей выборки: ', y_train.shape, '\n',
      'Размер для показателя тестовой выборки: ', y_test.shape)

plt.bar(target_names, data.target.value_counts(sort=False))
plt.xticks(target_names)
plt.ylim([100, 200])
plt.show()

# Логистическая регрессия

model = LogisticRegression(random_state=271)
model.fit(x_train, y_train)
y_predict = model.predict(x_test)
print('Предсказанные значения: \n', y_predict)
print('Исходные значения: \n', np.array(y_test))
print(classification_report(y_test, y_predict))

fig = px.imshow(confusion_matrix(y_test, y_predict), text_auto=True)
fig.update_layout(xaxis_title='Target', yaxis_title='Prediction')
fig.show()

# SVM

param_kernel = ('linear', 'rbf', "poly", "sigmod")
parameters = {'kernel': param_kernel}
model2 = SVC()
grid_search_svm = GridSearchCV(estimator=model2, param_grid=parameters, cv=6)
grid_search_svm.fit(x_train, y_train)

best_model = grid_search_svm.best_estimator_
print(best_model.kernel)
svm_preds = best_model.predict(x_test)
print(classification_report(svm_preds, y_test))

fig2 = px.imshow(confusion_matrix(svm_preds, y_test), text_auto=True)
fig2.update_layout(xaxis_title='Target', yaxis_title='Prediction')
fig2.show()

# KNN

number_of_neighbors = np.arange(3, 10, 25)
model3 = KNeighborsClassifier()
params = {"n_neighbors": number_of_neighbors}
grid_search = GridSearchCV(estimator=model3, param_grid=params, cv=6)
grid_search.fit(x_train, y_train)
print(grid_search.best_score_)
grid_search.best_estimator_

