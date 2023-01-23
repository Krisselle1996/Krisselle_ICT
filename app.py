from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    sl     = float(request.form['sepal_length'])
    sw     = float(request.form['sepal_width'])
    pl     = float(request.form['petal_length'])
    pw     = float(request.form['petal_width'])
    
   
    result = model.predict([[sl, sw, pl, pw]])
    result = result.item()
    result = round(result, 2)
        
    #prediction = svm_rbf.predict(result) 
    #predict_val = iris_data.target_names[prediction[0]]
    
    species_list = ['Iris_setosa', 'Iris_versicolor', 'Iris_virginica']
    species_name = species_list[result]
    return render_template('result.html', prediction_text="The Class of Iris Data is {}".format(species_name))
  
if __name__=='__main__':
    app.run(debug = True)