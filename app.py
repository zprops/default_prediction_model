# 🛑 Import the necessary libraries
from flask import Flask, request, jsonify
import joblib
import json
from discord import SyncWebhook
import pandas as pd
import numpy as np
import logging

app = Flask(__name__)

# 🛑 Create a function to load the model and other transformers
def load_sklearn_objects():
    # 👇 Add code to load the model and transformers using joblib.load()
    # Load other preprocessing components as needed
    # Add each loaded component to the sklearn_objects_dict dictionary and return it
    sklearn_objects_dict = {
        'pipeline': joblib.load('model_pipeline.pkl')
    }

    return sklearn_objects_dict

# 🛑 Convert the json requests into a DataFrame
def json_to_dataframe(json_data):
    columns = [
        'AMT_ANNUITY', 'AMT_CREDIT', 'AMT_GOODS_PRICE',
        'AMT_INCOME_TOTAL', 'CNT_CHILDREN', 'CNT_FAM_MEMBERS',
        'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_ID_PUBLISH',
        'DAYS_LAST_PHONE_CHANGE', 'DAYS_REGISTRATION',
        'OWN_CAR_AGE', 'CODE_GENDER', 'FLAG_EMAIL',
        'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'FLAG_PHONE',
        'FLAG_WORK_PHONE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS',
        'NAME_HOUSING_TYPE', 'NAME_INCOME_TYPE', 'OCCUPATION_TYPE',
    ]
    
    data_dict = {col: [json_data['data'][col]] for col in columns}
    dataframe = pd.DataFrame(data_dict)
    
    return dataframe

# 🛑 Preprocess the data in DataFrame format using sklearn_objects_dict and return it
def preprocess_data(df_request, sklearn_objects_dict):

    # 👇 Preprocess the input data using your exported preprocessing components
    # Example: scaled_data = sklearn_objects_dict["scaler"].transform(df_request[numeric_columns])
    
    # Changing specific values
    if pd.isnull(df_request.loc[0, 'OCCUPATION_TYPE']):
        df_request['OCCUPATION_TYPE'] = 'No_Type/Unknown'
    if pd.isnull(df_request.loc[0,'OWN_CAR_AGE']) and df_request.loc[0,'FLAG_OWN_CAR'] == 'N':
        df_request['OWN_CAR_AGE'] = -1
    if df_request.loc[0,'DAYS_EMPLOYED'] == 365243:
        df_request['DAYS_EMPLOYED'] = 1
    
    # Feature Engineering
    df_request['CNT_FAM_MEMBERS'] = np.where((df_request['CNT_FAM_MEMBERS'] - df_request['CNT_CHILDREN']) == 1.0, 1, 0)
    df_request['AMT_GOODS_PRICE'] = (df_request['AMT_CREDIT'] - df_request['AMT_GOODS_PRICE']) / df_request[['AMT_CREDIT', 'AMT_GOODS_PRICE']].mean(axis=1)
    df_request['AMT_ANNUITY'] = df_request['AMT_ANNUITY'] / df_request['AMT_INCOME_TOTAL']

    # Changing the Column Names
    df_request.rename(columns={
        'CNT_FAM_MEMBERS': 'FLAG_IS_SINGLE',
        'AMT_GOODS_PRICE': 'RATE_OVERHEAD_CREDIT',
        'AMT_ANNUITY': 'RATE_DEBT_SERVICE'
    }, inplace=True)

    return df_request

# 🛑 Get model prediction using the optimized threshold
def get_model_prediction(X_request, sklearn_objects_dict):
    # 👇 Make a prediction using your machine learning model and round it with the optimized threshold
    threshold = 0.72
    probas = sklearn_objects_dict['pipeline'].predict_proba(X_request)[:, 1]
    prediction = np.where(probas >= threshold, 1, 0)
    return prediction

# 🛑 Create a new function to handle incoming requests to your API
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 🛑 Get the input data as a JSON object
        request_json = request.get_json()

        # 🛑 Log the request fields
        logging.info(f"Request JSON: {request_json}")

        # 🛑 Convert the request into DataFrame format
        df_request = json_to_dataframe(json_data=request_json)

        # 🛑 Preprocess the request data
        X_request = preprocess_data(df_request=df_request, sklearn_objects_dict=sklearn_objects_dict)

        # 🛑Get model predictions using the optimized threshold
        prediction = get_model_prediction(X_request=X_request, sklearn_objects_dict=sklearn_objects_dict)

        # 🛑 Convert the predictions to a JSON object
        result = {
            'id': request_json['id'],
            'prediction': int(prediction) # we need to get the prediction from the numpy array and be sure that it's an int
        }

        # 🛑 Log the response
        logging.info(f"Response JSON: {result}")

        # 🛑Return the JSON object as a response
        return jsonify(result)
    
    # 🛑 Catch and handle any errors
    except Exception as e:
        error_msg = str(e)
        response = {
            'error': 'An error occurred during processing the request.',
            'message': error_msg
        }

        # 🛑 Convert the request_json into a readable string
        request_str = json.dumps(request_json, indent=4)
    
        # 🛑 Create the error message
        error_message = f"An error occurred during processing the request.\n\nRequest:\n{request_str}\n\nError:\n{error_msg}"
      
        # 🛑 Send the error message
        webhook.send(error_message)

        return make_response(jsonify(response), 500)

# 🛑 Add the following lines to start your Flask application when you run the script
if __name__ == "__main__":
    # 🛑 Create a logger to log requests and responses
    logging.basicConfig(level=logging.INFO)
    
    # 🛑 Load the model and transformers by calling the load_sklearn_objects function
    sklearn_objects_dict = load_sklearn_objects()

    # 👇 Store your Discord Webhook URL here
    webhook_url = 'https://discord.com/api/webhooks/1114973701398605925/zijH7KwflKEBqQSiG18v7NdZVlcIN0Q2xTlFSkcEBmczMlztQdw-30avdDQeEuATT8nz'

    # 🛑 Connect to Discord Webhook
    webhook = SyncWebhook.from_url(webhook_url)

    # 🛑 Start the application
    app.run(host="0.0.0.0", port=8000)
