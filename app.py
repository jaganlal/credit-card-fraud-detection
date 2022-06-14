import os, sys
from os.path import exists
from datetime import datetime

from flask import Flask, request, render_template
from flask_cors import CORS

import json
import pandas as pd
import numpy as np

import pickle
from zipfile import ZipFile

from sklearn import metrics, model_selection
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder

app = Flask(__name__)
cors = CORS(app, resources={r'/*': {'origins': '*'}})
app.config['DEBUG'] = True

# Globals
default_model_path = 'model'
default_model = '/fraud-detection.pkl'
default_model_zip = '/fraud-detection.zip'

# Routes
@app.route('/')
def main():
  extract_model_from_zip()
  return render_template('index.html')

@app.route('/health')
def health():
  return '<h3>Health Check OK!!</h3>'

@app.route('/predict', methods=['POST'])
def predict_fraudulent_transaction():
  '''Predict whether its a fraudulent transaction or not

  :param params: {
    data (required): 'contains the transaction data',
      transaction_time: '2020-06-21 12:14:25',
      credit_card_number: '2291163933867244',
      merchant: 'fraud_Kirlin and Sons',
      category: 'personal_care',
      amount: '2.86',
      first: 'Jeff',
      last: 'Elliott',
      gender: 'M',
      street: '351 Darlene Green',
      city: 'Columbia',
      state: 'SC',
      zip: 29209,
      lat: 33.9659,
      long: -80.9355,
      city_pop: 333497,
      job: 'Mechanical engineer',
      dob: '1968-03-19',
      transaction_id: '2da90c7d74bd46a0caf3777415b3ebd3',
      unix_time: 1371816865,
      merch_lat: 33.986391,
      merch_long: -81.200714
    model_name (optional): 'path and name of the model to load'
  }
  :return: success - result object with success message, 
           failure - result object with error message
  '''
  params = request.get_json()
  print('Prediction Params:', params)
  result = predict(params)
  return json.dumps(result)

def extract_model_from_zip():
  compressed_model_path = default_model_path + default_model_zip
  full_path = os.path.join(os.path.dirname(__file__), compressed_model_path)

  with ZipFile(full_path, 'r') as zipObj:
   # Extract all the contents of zip file in current directory
  #  extraction_path = os.path.join(os.path.dirname(__file__))
   zipObj.extractall()

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

    # check if the pickle file exists or not
    file_exists = exists(model_name)
    if file_exists == False:
      extract_model_from_zip()
      file_exists = exists(model_name)
      if file_exists == False:
        result = {
          'error': 'Model does not exist'
        }
        return result

    features = ['transaction_id', 'hour_of_day', 'category', 'amount(usd)', 'merchant', 'job']
    # df_test = pd.DataFrame(data.items())
    df_test = pd.DataFrame([data.values()],
                            columns=data.keys())

    # Apply function utcfromtimestamp and drop column unix_time
    df_test['time'] = df_test['unix_time'].apply(datetime.utcfromtimestamp)

    # Add cloumn hour of day
    df_test['hour_of_day'] = df_test.time.dt.hour

    df_test = df_test[features].set_index("transaction_id")
    enc = OrdinalEncoder(dtype=np.int64)
    enc.fit(df_test.loc[:, ['category','merchant','job']])

    df_test.loc[:, ['category','merchant','job']] = enc.transform(df_test[['category','merchant','job']])

    # Use pickle to load in the pre-trained model
    with open(model_name, 'rb') as f:
      model = pickle.load(f)

    prediction  = model.predict(df_test)
    probability = model.predict_proba(df_test)[:, 1]

    if prediction[0] == 1:
      status = 'This transaction seems to be Fraudulent, with a probability of {0}'.format(probability[0])
    else:
      status = 'Legitimate Transaction'
    result = {
      'Status': status
    }
    print('Status:', status)

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
#       "transaction_time": "2020-06-21 12:14:25",
#       "credit_card_number": "2291163933867244",
#       "merchant": "fraud_Kirlin and Sons",
#       "category": "personal_care",
#       "amount(usd)": "2.86",
#       "first": "Jeff",
#       "last": "Elliott",
#       "gender": "M",
#       "street": "351 Darlene Green",
#       "city": "Columbia",
#       "state": "SC",
#       "zip": 29209,
#       "lat": 33.9659,
#       "long": -80.9355,
#       "city_pop": 333497,
#       "job": "Mechanical engineer",
#       "dob": "1968-03-19",
#       "transaction_id": "2da90c7d74bd46a0caf3777415b3ebd3",
#       "unix_time": 1371816865,
#       "merch_lat": 33.986391,
#       "merch_long": -81.200714
#     }
# }

# Fraudulent Transaction
# {
#     "data": {
#       "transaction_time": "2020-06-21 22:32:22",
#       "credit_card_number": "6564459919350820",
#       "merchant": "fraud_Rodriguez, Yost and Jenkins",
#       "category": "misc_net",
#       "amount(usd)": "780.52",
#       "first": "Douglas",
#       "last": "Willis",
#       "gender": "M",
#       "street": "619 Jeremy Garden Apt. 681",
#       "city": "Benton",
#       "state": "WI",
#       "zip": 53803,
#       "lat": 42.5545,
#       "long": -90.3508,
#       "city_pop": 1306,
#       "job": "Public relations officer",
#       "dob": "1958-09-10",
#       "transaction_id": "ab4b379d2c0c9c667d46508d4e126d72",
#       "unix_time": 1371853942,
#       "merch_lat": 42.461127000000005,
#       "merch_long": -91.147148
#     }
# }