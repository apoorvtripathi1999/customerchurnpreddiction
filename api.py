from typing import Optional
from fastapi import FastAPI, UploadFile, File, Path, HTTPException
import pandas as pd
import cloudpickle
from fastapi.responses import JSONResponse
from io import StringIO
import os
import json
from pydantic import BaseModel
from datetime import date

app = FastAPI(title="Customer Churn Prediction",description="This API takes in user data and predicts weather the user will re-subscribe to the service or not", version='1.0.0')
json_path = "data/userdata.json"
user_path = "data/users.json"

# Performing validation using pydantic
class dataval(BaseModel):
   user_id: Optional[int] = None;	
   city: int;
   gender: str;
   registered_via: int;	
   payment_method_id: int;
   payment_plan_days: int;
   actual_amount_paid: int;	
   is_auto_renew: int;	
   transaction_date: date;
   membership_expire_date: date

# fuction for creating new users and validating existing users
def valid_user(user: int):
   if pd.isna(user):
      with open(user_path, "r") as f:
         data = json.load(f)
         max_user = max(data)
         user = max_user + 1
         data.append(int(user))
         with open(user_path, "w") as f:
            json.dump(data,f,indent=2)
         return user
   else:
       with open(user_path, "r") as f:
         data = json.load(f)
         if user not in data:
            data.append(int(user))
            with open(user_path, "w") as f:
               json.dump(data,f,indent=2)
       return user


# loading the model
try:
    with open("model/model.pickle","rb") as f:
     model = cloudpickle.load(f)
     print("Model Loaded successfully!!")
except Exception as e:
    print("Model could not be loaded, exception:")
    print(e)

# loading the pipeline
try:
    with open("model/pipe.pickle","rb") as f:
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


@app.get("/health")
def health():
   return JSONResponse(content={"status": "ok","Message":"API is Healthy"},status_code=200)


@app.post("/predict")
def predict(
   city: int,
   gender: str,
   registered_via: int,	
   payment_method_id: int,
   payment_plan_days: int,
   actual_amount_paid: int,	
   is_auto_renew: int,	
   transaction_date: date,
   membership_expire_date: date,
   user_id: Optional[int] = None	):
    
    result = {}

    data = dataval(
         user_id = user_id,
         city = city,
         gender = gender,
         registered_via =  registered_via,	
         payment_method_id = payment_method_id,
         payment_plan_days = payment_plan_days,
         actual_amount_paid = actual_amount_paid,
         is_auto_renew = is_auto_renew,
         transaction_date = transaction_date,
         membership_expire_date = membership_expire_date)
    
    user = valid_user(data.user_id)

    pipe_data = [[     	
         data.city,
         data.gender,
         data.registered_via,	
         data.payment_method_id,
         data.payment_plan_days,
         data.actual_amount_paid,	
         data.is_auto_renew,	
         data.transaction_date,
         data.membership_expire_date]]
   
    print(data)
    # transforming data using the pipeline and creating a trnaformed dataframe
    try:
      transformed = pipe.transform(pipe_data)
      df_transformed = pd.DataFrame(transformed, columns=["duration_of_subscription","female","male","city","registered_via","payment_method_id","payment_plan_days","actual_amount_paid","is_auto_renew"])
    except:
       raise HTTPException(status_code=500, detail="not able to tranform data through the pipeline")
    

    # Preforming prediction on the transformed data and updating the user details file
    try:
       df_transformed["predictions"] = model.predict(df_transformed)
       temp_result = df_transformed.iloc[0].to_dict()
       result[user] = temp_result
       if os.path.exists(json_path):
          try:
             with open(json_path, "r") as f:
               json_file = json.load(f)
          except:
             raise HTTPException(status_code=500, detail="Not able to load the json file")
       else:
          json_file = {}

       json_file.update(result)

       with open(json_path,"w") as f:
          json.dump(json_file, f, indent=2)

       return result
    except Exception as e:
       print(e)


@app.post("/bulkpredict")
def bulkpredict(file: UploadFile = File(...)):
      
      #reading the file
      content = file.file.read()
      
      #loading the file and converting it to dataframe
      try:
        df = pd.read_csv(StringIO(content.decode("utf-8")))
        print("data loaded successfully")
      except Exception as e:
         raise HTTPException(status_code=500, detail=f'CSV not loaded correctly: {e}')
      
      model_db = df.iloc[:,1:]
      users = df.iloc[:,0]

      # Creating new users
      users = users.to_numpy()
      for i in range(0,len(users)):
         users[i] = valid_user(users[i])
      users = users.astype(int)

      # fitting the dataframe to the pipeline
      try:
        trnsformed = pipe.transform(model_db)
        print("data transformed")
      except Exception as e:
        raise HTTPException(status_code=400, detail=f'data not transformed correctly: {e}')

      # pridicting the values and storing the predicted values in predict variable
      try:
        predict = model.predict(trnsformed)
        trans_df = pd.DataFrame(trnsformed, columns=["duration_of_subscription","female","male","city","registered_via","payment_method_id","payment_plan_days","actual_amount_paid","is_auto_renew"])
        trans_df["predictions"] = predict
        print("Predictions done")
      except Exception as e:
         raise HTTPException(status_code=500, detail=f'prediction not done correctly: {e}')
      final = {}

      # creating a key value pair JSON object(final result)
      try:
         trans_json = trans_df.to_dict(orient="records")
         for i in range(0,len(users)):
            final[str(users[i])] = trans_json[i]
      except Exception as e:
         print(f'Not able to generate the JSON result: {e}')
    
      # Saving and updating the json userdata file
      if os.path.exists(json_path):
         try:
           with open(json_path, "r") as f:
             data = json.load(f)
         except Exception as e:
            print(e)
            data = {}
      else:
         data = {}
      
      data.update(final)

      try:
         with open (json_path, "w") as f:
            json.dump(data,f, indent=2)
      except Exception as e:
         print(e)
      
      return final


@app.post("/getuser")
def getuser(target: int):
      
      # Opening the json file
      if os.path.exists(json_path):
         try:
           with open(json_path, "r") as f:
             data = json.load(f)
             print("data loaded successfully")
         except:
            raise HTTPException(status_code=500, detail="not able to read the json file")
      else:
         raise HTTPException(status_code=500, detail="Json file not read correctly")

      # searching for the target key and returning the result as a dictionary
      try:
          user_data = data.get(str(target))
          if user_data is None:
             raise HTTPException(status_code=500,detail="Not able to fetch user data")
          return {"user_id": target, "user_data": user_data}
      except:
         raise HTTPException(status_code=500, detail="Not able to fetch the list")
    

@app.post("/deleteuser")
def deleteuser(target: int):
    # Opening the json file
      if os.path.exists(json_path):
         try:
           with open(json_path, "r") as f:
             data = json.load(f)
         except:
            raise HTTPException(status_code=500, detail="not able to load the json file")
      else:
         raise HTTPException(status_code=500, detail="json path does not exist")
      
      if os.path.exists(user_path):
         try:
            with open(user_path,"r") as f:
               users = json.load(f)
         except:
            raise HTTPException(status_code=500, detail="Could not load users")
      else:
         raise HTTPException(status_code=500, detail="user file path does not exist")
      
      # searching for the target key and deleting it from the JSON file 
      try:
          
         # deleting the user from the users file
          users = [u for u in users if str(u) != str(target)]
         
         # deleting the user from the userdata file
          key = str(target)
          if key not in data:
             raise HTTPException(status_code=500, detail="the given user id does not exist")
          del data[key]
          
          # dumping updated json files
          with open(user_path,"w") as f:
             json.dump(users,f,indent=2)

          with open(json_path, "w") as f:
             json.dump(data, f, indent=2)  
         
          return {"message": f'{target} deleted successfully'}
      except:
         raise HTTPException(status_code=500, detail="Not able to fetch the list")

