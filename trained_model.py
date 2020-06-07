#lOAD DATA
from pycaret.datasets import get_data
data = get_data('insurance')
data

#SHOW EXPERIMENT 1
from pycaret.regression import *
s = setup(data, target = 'charges', session_id = 123)

#CREATE LOGISTIC REGRESSION MODEL
lr = create_model('lr')

#PLOT LR MODEL
plot_model(lr)

#EXPLORE EXPERIMENT 2
s2 = setup(data, target = 'charges', session_id = 123,
           normalize = True,
           polynomial_features = True, trigonometry_features = True, feature_interaction=True, 
           bin_numeric_features= ['age', 'bmi'])

#SHOW COLUMNS IN EXPERIMENT 2
s2[0].columns

#CREATE EXPERIMENT 2 MODEL
lr = create_model('lr')

#PLOT THE MODEL
plot_model(lr)

#save the model
save_model(lr, 'deployment_07062020')

#THEN LOAD THE SAVED MODEL
deployment_07062020 = load_model('deployment_07062020')
deployment_07062020

import requests
url = 'https://billprediction-insurance.herokuapp.com/predict_api'
pred = requests.post(url,json={'age':55, 'sex':'male', 'bmi':59, 'children':1, 'smoker':'male', 'region':'northwest'})
print(pred.json())
