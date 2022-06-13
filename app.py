import os, sys

from flask import Flask, request, render_template
from flask_cors import CORS

import json
import pandas as pd
import numpy as np

import pickle

from sklearn import metrics, model_selection
from sklearn.model_selection import train_test_split

from xgboost.sklearn import XGBClassifier

app = Flask(__name__)
cors = CORS(app, resources={r'/*': {'origins': '*'}})
app.config['DEBUG'] = True

# Globals
default_model_path = 'model'
default_model = '/fraud-detection.pkl'

# Routes
@app.route('/')
def main():
	return render_template('index.html')

@app.route('/health')
def health():
    return '<h3>Health Check OK!!</h3>'

@app.route('/predict', methods=['POST'])
def predict_from_model():
  '''Predict whether to approve loan or not
  :param params: {
    data (required): 'contains the data to predict the loan approval',
      Gender: 'Male/Female',
      Married: 'Yes/No',
      Dependents: '0/1/2/3+',
      Education: 'Graduate/Not Graduate',
      Self_Employed: 'Yes/No',
      ApplicantIncome: Numeric value,
      CoapplicantIncome: Numeric value,
      LoanAmount: Numeric value,
      Loan_Amount_Term: Numeric value,
      Credit_History: Numeric value,
      Property_Area: 'Rural/Semiurban/Urban'
    model_name (optional): 'path and name of the model to load'
  }
  :return: success - result object with success message, 
           failure - result object with error message
  '''
  params = request.get_json()
  print('Prediction Params:', params)
  result = predict(params)
  return json.dumps(result)

def predict(params):
  result = {}
  data = None
  model_name = default_model_path + default_model

  try:
    if params:
      if params.get('data'):
        data = params.get('data')
      if params.get('model_name'):
        model_name = params.get('model_name')

    model_name = os.path.join(os.path.dirname(__file__), model_name)

    if isinstance(data['Gender'], str) and data['Gender'] is not None and len(data['Gender']) > 0:
      if (data['Gender'] == 'Male'):
        data['Gender'] = 1
      else:
        data['Gender'] = 0

    if isinstance(data['Education'], str) and data['Education'] is not None and len(data['Education']) > 0:
      if (data['Education'] == 'Graduate'):
        data['Education'] = 1
      else:
        data['Education'] = 0

    if isinstance(data['Married'], str) and data['Married'] is not None and len(data['Married']) > 0:
      if (data['Married'] == 'Yes'):
        data['Married'] = 1
      else:
        data['Married'] = 0

    if isinstance(data['Self_Employed'], str) and data['Self_Employed'] is not None and len(data['Self_Employed']) > 0:
      if (data['Self_Employed'] == 'Yes'):
        data['Self_Employed'] = 1
      else:
        data['Self_Employed'] = 0

    if isinstance(data['Dependents'], str) and data['Dependents'] is not None and len(data['Dependents']) > 0:
      if (data['Dependents'] == '0'):
        data['Dependents'] = 0
      elif (data['Dependents'] == '1'):
        data['Dependents'] = 1
      elif (data['Dependents'] == '2'):
        data['Dependents'] = 2
      elif (data['Dependents'] == '3+'):
        data['Dependents'] = 3
      else:
        data['Dependents'] = 0

    if isinstance(data['Property_Area'], str) and data['Property_Area'] is not None and len(data['Property_Area']) > 0:
      if (data['Property_Area'] == 'Rural'):
        data['Property_Area'] = 0
      elif (data['Property_Area'] == 'Semiurban'):
        data['Property_Area'] = 1
      elif (data['Property_Area'] == 'Urban'):
        data['Property_Area'] = 2
      else:
        data['Property_Area'] = 0
    
    data['LoanAmount_log'] = np.log(data['LoanAmount'])
    data['Total_Income_log'] = np.log(data['ApplicantIncome'] + data['CoapplicantIncome'])

    data.pop('ApplicantIncome')
    data.pop('CoapplicantIncome')
    data.pop('LoanAmount')

    # Use pickle to load in the pre-trained model
    with open(model_name, 'rb') as f:
      model = pickle.load(f)

    input_variables = pd.DataFrame([data.values()],
                                  columns=data.keys(),
                                  dtype=float)


    input_variables.sort_index(axis=1, inplace=True)
    prediction = model.predict(input_variables.head(1))
    
    loan_status = 'Denied'
    if(prediction[0] == 1):
      loan_status = 'Approved'

    result = {
      'Loan Status': loan_status
    }
    print('Loan Prediction:', prediction[0])

  except Exception as e:
    print('Exception in predict: {0}'.format(e))
    exc_type, exc_obj, exc_tb = sys.exc_info()
    fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
    print(exc_type, fname, exc_tb.tb_lineno)

    result = {
      'error': 'Exception in predict: {0}'.format(e)
    }

  return result

def is_integer_num(n):
    if isinstance(n, int):
        return True
    if isinstance(n, float):
        return n.is_integer()
    return False

if __name__ == '__main__':
  try:
    app.run(host='0.0.0.0', port=5050, debug=True)
  finally:
    print ('Closing flask server')

# TEST DATA FROM POSTMAN
# {
#     "data": {
#         "Gender": 1,
#         "Married": 1,
#         "Dependents": 0,
#         "Education": 1,
#         "Self_Employed": 0,
#         "ApplicantIncome": 5720,
#         "CoapplicantIncome": 0,
#         "LoanAmount": 110,
#         "Loan_Amount_Term": 360,
#         "Credit_History": 1,
#         "Property_Area": 2
#     }
# }
# {
#     "data": {
#         "Gender": "Male",
#         "Married": "Yes",
#         "Dependents": 0,
#         "Education": "Graduate",
#         "Self_Employed": "No",
#         "ApplicantIncome": 5720,
#         "CoapplicantIncome": 0,
#         "LoanAmount": 110,
#         "Loan_Amount_Term": 360,
#         "Credit_History": 1,
#         "Property_Area": "Urban"
#     }
# }