from typing import Union
from fastapi import FastAPI
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
import cloudpickle

app = FastAPI(title="Customer Churn Prediction",description="This API takes in user data and predicts weather the user will re-subscribe to the service or not", version='1.0.0')

# health of the api
# predict
# bulk predict
# model information


try:
    with open("customer-churn-prediction/model/model.pickle","rb") as f:
     model = cloudpickle.load(f)
     print("Model Loaded successfully!!")
except Exception as e:
    print("Model could not be loaded, exception:")
    print(e)

try:
    with open("customer-churn-prediction/model/pipe.pickle","rb") as f:
     pipe = cloudpickle.load(f)
     print("Pipeline Loaded successfully!!")
except Exception as e:
    print("Pipeline could not be loaded, exception:")
    print(e)

@app.get("/")
def root():
    return {"Name":"user Churn Prediction", 
            "Description": "This API takes in user data and predicts weather the user is going to renew thier membership or not",
            "Endpoints":{
                "/health": "Returns the current health of the api",
                "/predict": {
                    "Description" : "Takes user data for a single user and returs if they will re-subscribe to the service",
                    "Parameters:" : {
                        "duration_of_subscription" : "the duration of the existing subscription",	
                        "gender" : "gender of the user",	
                        "city" : "city of residense",	
                        "registered_via" : "registeration medium",	
                        "payment_method_id" : "the payment methord of the user",	
                        "payment_plan_days" : "the current payment plan of the user",	
                        "actual_amount_paid" : "ammount the user paid for the subscription",	
                        "is_auto_renew" : "weather the service is auto-renewable"
                    }
                },
                "/bulkpredict": {
                    "Description" : "Takes user data for a bulk of users and returs if they will re-subscribe to the service",
                    "Parameters:" : {
                        "duration_of_subscription" : "the duration of the existing subscription",	
                        "gender" : "gender of the user",	
                        "city" : "city of residense",	
                        "registered_via" : "registeration medium",	
                        "payment_method_id" : "the payment methord of the user",	
                        "payment_plan_days" : "the current payment plan of the user",	
                        "actual_amount_paid" : "ammount the user paid for the subscription",	
                        "is_auto_renew" : "weather the service is auto-renewable"
                    }
                },
                "/modelinfo": "returns the information about the model used"
            }
            }


@app.post("/predict")
def predict(duration_of_subscription: int, gender: str, city: int, registered_via: int, payment_method_id: int, payment_plan_days: int, actual_amount_paid: int, is_auto_renew: int):
    if gender == "male" or gender == "Male":
       male = 1
       female =0
    else:
       male = 0
       female = 1
    
    data = [duration_of_subscription,female,male,city,registered_via,payment_method_id,payment_plan_days,actual_amount_paid,is_auto_renew]
    print(data)

    try:
       prediction = model.predict([data])
       return {"Predicted Value": str(prediction[0])}
    
    except Exception as e:
       print(e)




