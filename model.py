# Importing the libraries

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import json

# Importing the dataset
iris_data = pd.read_excel('iris_flask.xls')

# Assigning independent and target variable
x = iris_data.drop('Classification', axis=1)
y = iris_data['Classification']

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 42)

# Model Training
from sklearn.svm import SVC
svm_rbf       = SVC(kernel = "rbf")
svm_model_rbf = svm_rbf.fit(x_train, y_train)



# Predicting the Test set results
#y_pred_rbf    = svm_model_rbf.predict(x_test)

# Saving model using pickle
pickle.dump(svm_rbf, open('model.pkl','wb'))

# Loading model to compare the results
#model = pickle.load( open('model.pkl','rb'))
#print(model.predict([[2.8]]))
