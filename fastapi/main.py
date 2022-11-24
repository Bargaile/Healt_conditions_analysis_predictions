import joblib
import uvicorn, os
import pandas as pd
from fastapi import FastAPI, HTTPException
from classes import *


app = FastAPI()

stroke_predictor = joblib.load("stroke_predictor.sav")
hypertension_predictor = joblib.load("hypertension_predictor.sav")
glucose_predictor = joblib.load("glucose_level_predictor.sav")
bmi_predictor = joblib.load("bmi_level_predictor.sav")
glucose_bmi_predictor = joblib.load("glucose_bmi_predictor.sav")
hypertension_bmi_predictor = joblib.load("hypertension_bmi_predictor.sav") 
hypertension_glucose_predictor = joblib.load("hypertension_glucose_predictor.sav") 
hyp_gluc_bmi_predictor = joblib.load("hypertension_glucose_bmi_predictor.sav") 


@app.get("/")
def home():
    return {"text": "Health predictions"}

@app.post("/stroke_prediction")
async def create_application(stroke_pred: stroke_prediction):

    stroke_df = pd.DataFrame()

    if stroke_pred.gender not in gender_dict:
        raise HTTPException(status_code=404, detail="Gender not found")

    if stroke_pred.hypertension not in hypertension_dict:
        raise HTTPException(status_code=404, detail="Hypertension disease not defined")
    
    if stroke_pred.heart_disease not in heart_disease_dict:
        raise HTTPException(status_code=404, detail="Heart disease unknown")
    
    if stroke_pred.ever_married not in ever_married_dict:
        raise HTTPException(status_code=404, detail="Marital status unknown")
    
    if stroke_pred.work_type not in work_type_dict:
        raise HTTPException(status_code=404, detail="Work situation unknown")
    
    if stroke_pred.residence_type not in residence_type_dict:
        raise HTTPException(status_code=404, detail="Residence type not found")
    
    if stroke_pred.smoking_status not in smoking_status_dict:
        raise HTTPException(status_code=404, detail="Smoking status not found")
    
    stroke_df["gender"] = [stroke_pred.gender]
    stroke_df["age"] = [stroke_pred.age]
    stroke_df["hypertension"] = [stroke_pred.hypertension]
    stroke_df["heart_disease"] = [stroke_pred.heart_disease]
    stroke_df["ever_married"]= [stroke_pred.ever_married]
    stroke_df["work_type"] = [stroke_pred.work_type]
    stroke_df["residence_type"] = [stroke_pred.residence_type]
    stroke_df["avg_glucose_level"] = [stroke_pred.avg_glucose_level]
    stroke_df["bmi"] = [stroke_pred.bmi]
    stroke_df["smoking_status"] = [stroke_pred.smoking_status]

    prediction = stroke_predictor.predict(stroke_df)
    if prediction[0] == 0:
        prediction = "You are not likely to get a stroke."
    else:
        prediction = "You are in the danger zone of stroke. Please contact your family doctor."

    return {'prediction': prediction}

@app.post("/hypertension_prediction")
async def create_application(hyp_pred: hypertension_prediction):

    hypertension_df = pd.DataFrame()


    if hyp_pred.gender not in gender_dict:
        raise HTTPException(status_code=404, detail="gender not found")

    if hyp_pred.heart_disease not in heart_disease_dict:
        raise HTTPException(status_code=404, detail="Heart disease unknown")
    
    if hyp_pred.ever_married not in ever_married_dict:
        raise HTTPException(status_code=404, detail="Marital status unknown")
    
    if hyp_pred.work_type not in work_type_dict:
        raise HTTPException(status_code=404, detail="Work situation unknown")
    
    if hyp_pred.residence_type not in residence_type_dict:
        raise HTTPException(status_code=404, detail="Residence type not found")
    
    if hyp_pred.smoking_status not in smoking_status_dict:
        raise HTTPException(status_code=404, detail="Smoking status not found")

    hypertension_df["gender"] = [hyp_pred.gender]
    hypertension_df["age"] = [hyp_pred.age]
    hypertension_df["heart_disease"] = [hyp_pred.heart_disease]
    hypertension_df["ever_married"]= [hyp_pred.ever_married]
    hypertension_df["work_type"] = [hyp_pred.work_type]
    hypertension_df["residence_type"] = [hyp_pred.residence_type]
    hypertension_df["avg_glucose_level"] = [hyp_pred.avg_glucose_level]
    hypertension_df["bmi"] = [hyp_pred.bmi]
    hypertension_df["smoking_status"] = [hyp_pred.smoking_status]

    prediction = hypertension_predictor.predict(hypertension_df)
    if prediction[0] == 0:
        prediction = "You are not likely to get have hypertension."
    else:
        prediction = "You are in the danger zone of hypertension. Please contact your family doctor."

    return {'prediction': prediction}


@app.post("/glucose_level_prediction")
async def create_application(gluc_level_pred: glucose_prediction):


    glucose_df = pd.DataFrame()

    if gluc_level_pred.gender not in gender_dict:
        raise HTTPException(status_code=404, detail="gender not found")

    if gluc_level_pred.hypertension not in hypertension_dict:
        raise HTTPException(status_code=404, detail="Hypertension disease not defined")
    
    if gluc_level_pred.heart_disease not in heart_disease_dict:
        raise HTTPException(status_code=404, detail="Heart disease unknown")
    
    if gluc_level_pred.ever_married not in ever_married_dict:
        raise HTTPException(status_code=404, detail="Marital status unknown")
    
    if gluc_level_pred.work_type not in work_type_dict:
        raise HTTPException(status_code=404, detail="Work situation unknown")
    
    if gluc_level_pred.residence_type not in residence_type_dict:
        raise HTTPException(status_code=404, detail="Residence type not found")
    
    if gluc_level_pred.smoking_status not in smoking_status_dict:
        raise HTTPException(status_code=404, detail="Smoking status not found")
    
    if gluc_level_pred.heart_hypertenz not in heart_hypertenz_dict:
        raise HTTPException(status_code=404, detail="Heart or hypertension diseases are not found")

    glucose_df["gender"] = [gluc_level_pred.gender]
    glucose_df["age"] = [gluc_level_pred.age]
    glucose_df["hypertension"] = [gluc_level_pred.hypertension]
    glucose_df["heart_disease"] = [gluc_level_pred.heart_disease]
    glucose_df["ever_married"]= [gluc_level_pred.ever_married]
    glucose_df["work_type"] = [gluc_level_pred.work_type]
    glucose_df["residence_type"] = [gluc_level_pred.residence_type]
    glucose_df["bmi"] = [gluc_level_pred.bmi]
    glucose_df["smoking_status"] = [gluc_level_pred.smoking_status]
    glucose_df["heart_hypertenz"] = [gluc_level_pred.heart_hypertenz]

    prediction = glucose_predictor.predict(glucose_df)
    if prediction[0] == 0:
        prediction = "Your glucose level is likely to be in the range of: 51-100 mg/dl"
    elif prediction[0] == 1:
        prediction = "Your glucose level is likely to be in the range of: 101-150 mg/dl"
    elif prediction[0] == 2:
        prediction = "Your glucose level is likely to be in the range of: 151-200 mg/dl"
    elif prediction[0] == 3:
        prediction = "Your glucose level is likely to be in the range of: 201-250 mg/dl"    
    else:
        prediction = "You are in the danger zone of glucose level (over 250 mg/dl). Please contact your family doctor."

    return {'prediction': prediction}


@app.post("/bmi_group_prediction")
async def create_application(bmi_group_pred: bmi_prediction):

    bmi_df = pd.DataFrame()

    if bmi_group_pred.gender not in gender_dict:
        raise HTTPException(status_code=404, detail="Gender not found")

    if bmi_group_pred.hypertension not in hypertension_dict:
        raise HTTPException(status_code=404, detail="Hypertension disease not defined")
    
    if bmi_group_pred.heart_disease not in heart_disease_dict:
        raise HTTPException(status_code=404, detail="Heart disease unknown")
    
    if bmi_group_pred.ever_married not in ever_married_dict:
        raise HTTPException(status_code=404, detail="Marital status unknown")
    
    if bmi_group_pred.work_type not in work_type_dict:
        raise HTTPException(status_code=404, detail="Work situation unknown")
    
    if bmi_group_pred.residence_type not in residence_type_dict:
        raise HTTPException(status_code=404, detail="Residence type not found")
    
    if bmi_group_pred.smoking_status not in smoking_status_dict:
        raise HTTPException(status_code=404, detail="Smoking status not found")
    
    if bmi_group_pred.heart_hypertenz not in heart_hypertenz_dict:
        raise HTTPException(status_code=404, detail="Heart or hypertension diseases are not found")

    bmi_df["gender"] = [bmi_group_pred.gender]
    bmi_df["age"] = [bmi_group_pred.age]
    bmi_df["hypertension"] = [bmi_group_pred.hypertension]
    bmi_df["heart_disease"] = [bmi_group_pred.heart_disease]
    bmi_df["ever_married"]= [bmi_group_pred.ever_married]
    bmi_df["work_type"] = [bmi_group_pred.work_type]
    bmi_df["residence_type"] = [bmi_group_pred.residence_type]
    bmi_df["avg_glucose_level"] = [bmi_group_pred.avg_glucose_level]
    bmi_df["smoking_status"] = [bmi_group_pred.smoking_status]
    bmi_df["heart_hypertenz"] = [bmi_group_pred.heart_hypertenz]

    prediction = bmi_predictor.predict(bmi_df)
    if prediction[0] == 0:
        prediction = "Your are under weighted, your bmi is <=17"
    elif prediction[0] == 1:
        prediction = "Your bmi is in the normal range of 18-24"
    elif prediction[0] == 2:
        prediction = "Your bmi is in the range of slightly overweight: 25-30"
    elif prediction[0] == 3:
        prediction = "Your bmi is in the overweight range of 31-35"    
    elif prediction[0] == 4:
        prediction = "Your bmi is in the medium overweight range of 35-40"   
    else:
        prediction = "You have overweight, your bmi is over 41. Please contact your family doctor."

    return {'prediction': prediction}


@app.post("/glucose_bmi_prediction")
async def create_application(gluc_bmi_pred: glucose_bmi_prediction):

    gluc_bmi_df = pd.DataFrame()

    if gluc_bmi_pred.gender not in gender_dict:
        raise HTTPException(status_code=404, detail="Gender not found")

    if gluc_bmi_pred.hypertension not in hypertension_dict:
        raise HTTPException(status_code=404, detail="Hypertension disease not defined")
    
    if gluc_bmi_pred.heart_disease not in heart_disease_dict:
        raise HTTPException(status_code=404, detail="Heart disease unknown")
    
    if gluc_bmi_pred.ever_married not in ever_married_dict:
        raise HTTPException(status_code=404, detail="Marital status unknown")
    
    if gluc_bmi_pred.work_type not in work_type_dict:
        raise HTTPException(status_code=404, detail="Work situation unknown")
    
    if gluc_bmi_pred.residence_type not in residence_type_dict:
        raise HTTPException(status_code=404, detail="Residence type not found")
    
    if gluc_bmi_pred.smoking_status not in smoking_status_dict:
        raise HTTPException(status_code=404, detail="Smoking status not found")
    
    if gluc_bmi_pred.heart_hypertenz not in heart_hypertenz_dict:
        raise HTTPException(status_code=404, detail="Heart or hypertension diseases are not found")

    gluc_bmi_df["gender"] = [gluc_bmi_pred.gender]
    gluc_bmi_df["age"] = [gluc_bmi_pred.age]
    gluc_bmi_df["hypertension"] = [gluc_bmi_pred.hypertension]
    gluc_bmi_df["heart_disease"] = [gluc_bmi_pred.heart_disease]
    gluc_bmi_df["ever_married"]= [gluc_bmi_pred.ever_married]
    gluc_bmi_df["work_type"] = [gluc_bmi_pred.work_type]
    gluc_bmi_df["residence_type"] = [gluc_bmi_pred.residence_type]
    gluc_bmi_df["smoking_status"] = [gluc_bmi_pred.smoking_status]
    gluc_bmi_df["heart_hypertenz"] = [gluc_bmi_pred.heart_hypertenz]

    prediction = glucose_bmi_predictor.predict(gluc_bmi_df)
    
    if prediction[0][0] ==0 and prediction[0][1] == 0:
        prediction = "Your glucose level is likely to be in the range of: 51-100 mg/dl. Your are under weighted, your bmi is <=17."
    elif prediction[0][0] == 1 and prediction[0][1] == 0:
        prediction = "Your glucose level is likely to be in the range of: 101-150 mg/dl.Your are under weighted, your bmi is <=17."
    elif prediction[0][0] == 2 and prediction[0][1] == 0:
        prediction = "Your glucose level is likely to be in the range of: 151-200 mg/dl.Your are under weighted, your bmi is <=17."
    elif prediction[0][0] == 3 and prediction[0][1] == 0:
        prediction = "Your glucose level is likely to be in the range of: 201-250 mg/dl.Your are under weighted, your bmi is <=17."
    elif prediction[0][0] == 4 and prediction[0][1] == 0:
        prediction = "You are in the danger zone of glucose level (over 250 mg/dl).Your are under weighted, your bmi is <=17."
    elif prediction[0][0] == 0 and prediction[0][1] == 1:
        prediction = "Your glucose level is likely to be in the range of: 51-100 mg/dl. Your bmi is in the normal range of 18-24"
    elif prediction[0][0]==1 and prediction[0][1] == 1:
        prediction = "Your glucose level is likely to be in the range of: 101-150 mg/dl.Your bmi is in the normal range of 18-24"
    elif prediction[0][0] == 2 and prediction[0][1] == 1:
        prediction = "Your glucose level is likely to be in the range of: 151-200 mg/dl.Your bmi is in the normal range of 18-24"
    elif prediction[0][0] == 3 and prediction[0][1] == 1:
        prediction = "Your glucose level is likely to be in the range of: 201-250 mg/dl.Your bmi is in the normal range of 18-24"
    elif prediction[0][0] == 4 and prediction[0][1] == 1:
        prediction = "You are in the danger zone of glucose level (over 250 mg/dl).Your bmi is in the normal range of 18-24"
    elif prediction[0][0] == 0 and prediction[0][1] == 2:
        prediction = "Your glucose level is likely to be in the range of: 51-100 mg/dl. Your bmi is in the range of slightly overweight: 25-30"
    elif prediction[0][0] == 1 and prediction[0][1] == 2:
        prediction = "Your glucose level is likely to be in the range of: 101-150 mg/dl.Your bmi is in the range of slightly overweight: 25-30"
    elif prediction[0][0] == 2 and prediction[0][1] == 2:
        prediction = "Your glucose level is likely to be in the range of: 151-200 mg/dl.Your bmi is in the range of slightly overweight: 25-30"
    elif prediction[0][0] == 3 and prediction[0][1] == 2:
        prediction = "Your glucose level is likely to be in the range of: 201-250 mg/dl.Your bmi is in the range of slightly overweight: 25-30"
    elif prediction[0][0] == 4 and prediction[0][1] == 2:
        prediction = "You are in the danger zone of glucose level (over 250 mg/dl).Your bmi is in the range of slightly overweight: 25-30"
    elif prediction[0][0] == 0 and prediction[0][1] == 3:
        prediction = "Your glucose level is likely to be in the range of: 51-100 mg/dl. Your bmi is in the overweight range of 31-35"
    elif prediction[0][0] == 1 and prediction[0][1] == 3:
        prediction = "Your glucose level is likely to be in the range of: 101-150 mg/dl.Your bmi is in the overweight range of 31-35"
    elif prediction[0][0] == 2 and prediction[0][1] == 3:
        prediction = "Your glucose level is likely to be in the range of: 151-200 mg/dl.Your bmi is in the overweight range of 31-35"
    elif prediction[0][0] == 3 and prediction[0][1] == 3:
        prediction = "Your glucose level is likely to be in the range of: 201-250 mg/dl.Your bmi is in the overweight range of 31-35"
    elif prediction[0][0] == 4 and prediction[0][1] == 3:
        prediction = "You are in the danger zone of glucose level (over 250 mg/dl).Your bmi is in the overweight range of 31-35"
    elif prediction[0][0] == 0 and prediction[0][1] == 4:
        prediction = "Your glucose level is likely to be in the range of: 51-100 mg/dl. Your bmi is in the medium overweight range of 35-40"
    elif prediction[0][0] == 1 and prediction[0][1] == 4:
        prediction = "Your glucose level is likely to be in the range of: 101-150 mg/dl.Your bmi is in the medium overweight range of 35-40"
    elif prediction[0][0] == 2 and prediction[0][1] == 4:
        prediction = "Your glucose level is likely to be in the range of: 151-200 mg/dl.Your bmi is in the medium overweight range of 35-40"
    elif prediction[0][0] == 3 and prediction[0][1] == 4:
        prediction = "Your glucose level is likely to be in the range of: 201-250 mg/dl.Your bmi is in the medium overweight range of 35-40"
    elif prediction[0][0] == 4 and prediction[0][1] == 4:
        prediction = "You are in the danger zone of glucose level (over 250 mg/dl).Your bmi is in the medium overweight range of 35-40"
    elif prediction[0][0] == 0 and prediction[0][1] == 5:
        prediction = "Your glucose level is likely to be in the range of: 51-100 mg/dl.You have overweight, your bmi is over 41. Please contact your family doctor."
    elif prediction[0][0] == 1 and prediction[0][1] == 5:
        prediction = "Your glucose level is likely to be in the range of: 101-150 mg/dl.You have overweight, your bmi is over 41. Please contact your family doctor."
    elif prediction[0][0] == 2 and prediction[0][1] == 5:
        prediction = "Your glucose level is likely to be in the range of: 151-200 mg/dl.You have overweight, your bmi is over 41. Please contact your family doctor."
    elif prediction[0][0] == 3 and prediction[0][1] == 5:
        prediction = "Your glucose level is likely to be in the range of: 201-250 mg/dl.You have overweight, your bmi is over 41. Please contact your family doctor."
    elif prediction[0][0] == 4 and prediction[0][1] == 5:
        prediction = "You are in the danger zone of glucose level (over 250 mg/dl).You have overweight, your bmi is over 41. Please contact your family doctor."
    return {'prediction': prediction}

@app.post("/hypertension_bmi_prediction")
async def create_application(hyp_bmi_pred: hypertension_bmi_prediction):

    hyp_bmi_df = pd.DataFrame()

    if hyp_bmi_pred.gender not in gender_dict:
        raise HTTPException(status_code=404, detail="Gender not found")

    if hyp_bmi_pred.heart_disease not in heart_disease_dict:
        raise HTTPException(status_code=404, detail="Heart disease unknown")
    
    if hyp_bmi_pred.ever_married not in ever_married_dict:
        raise HTTPException(status_code=404, detail="Marital status unknown")
    
    if hyp_bmi_pred.work_type not in work_type_dict:
        raise HTTPException(status_code=404, detail="Work situation unknown")
    
    if hyp_bmi_pred.residence_type not in residence_type_dict:
        raise HTTPException(status_code=404, detail="Residence type not found")
    
    if hyp_bmi_pred.smoking_status not in smoking_status_dict:
        raise HTTPException(status_code=404, detail="Smoking status not found")

    hyp_bmi_df["gender"] = [hyp_bmi_pred.gender]
    hyp_bmi_df["age"] = [hyp_bmi_pred.age]
    hyp_bmi_df["heart_disease"] = [hyp_bmi_pred.heart_disease]
    hyp_bmi_df["ever_married"]= [hyp_bmi_pred.ever_married]
    hyp_bmi_df["work_type"] = [hyp_bmi_pred.work_type]
    hyp_bmi_df["residence_type"] = [hyp_bmi_pred.residence_type]
    hyp_bmi_df["avg_glucose_level"] = [hyp_bmi_pred.avg_glucose_level]
    hyp_bmi_df["smoking_status"] = [hyp_bmi_pred.smoking_status]

    prediction = hypertension_bmi_predictor.predict(hyp_bmi_df)
    if prediction[0][0] == 0 and prediction[0][1] == 0:
        prediction = "You do not have hypertension. Your are under weighted, your bmi is <=17"
    elif prediction[0][0] == 0 and prediction[0][1] == 1:
        prediction = "You do not have hypertension.Your bmi is in the normal range of 18-24"
    elif prediction[0][0] == 0 and prediction[0][1] == 2:
        prediction = "You do not have hypertension.Your bmi is in the range of slightly overweight: 25-30"
    elif prediction[0][0] == 0 and prediction[0][1] == 3:
        prediction = "You do not have hypertension.Your bmi is in the overweight range of 31-35"    
    elif prediction[0][0] == 0 and prediction[0][1] == 4:
        prediction = "You do not have hypertension.Your bmi is in the medium overweight range of 35-40"   
    elif prediction[0][0] == 0 and prediction[0][1] == 5:
        prediction = "You do not have hypertension.You have overweight, your bmi is over 41. Please contact your family doctor."
    if prediction[0][0] == 1 and prediction[0][1] == 0:
        prediction = "You are in the risk one to have hypertension. Your are under weighted, your bmi is <=17"
    elif prediction[0][0] == 1 and prediction[0][1] == 1:
        prediction = "You are in the risk one to have hypertension.Your bmi is in the normal range of 18-24"
    elif prediction[0][0] == 1 and prediction[0][1] == 2:
        prediction = "You are in the risk one to have hypertension.Your bmi is in the range of slightly overweight: 25-30"
    elif prediction[0][0] == 1 and prediction[0][1] == 3:
        prediction = "You are in the risk one to have hypertension.Your bmi is in the overweight range of 31-35"    
    elif prediction[0][0] == 1 and prediction[0][1] == 4:
        prediction = "You are in the risk one to have hypertension.Your bmi is in the medium overweight range of 35-40"   
    elif prediction[0][0] == 1 and prediction[0][1] == 5:
        prediction = "You are in the risk one to have hypertension.You have overweight, your bmi is over 41. Please contact your family doctor."
    
    return {'prediction': prediction}


@app.post("/hypertension_glucose_prediction")
async def create_application(hyp_gluc_pred: hypertension_glucose_prediction):

    hyp_gluc_df = pd.DataFrame()

    if hyp_gluc_pred.gender not in gender_dict:
        raise HTTPException(status_code=404, detail="Gender not found")

    if hyp_gluc_pred.heart_disease not in heart_disease_dict:
        raise HTTPException(status_code=404, detail="Heart disease unknown")
    
    if hyp_gluc_pred.ever_married not in ever_married_dict:
        raise HTTPException(status_code=404, detail="Marital status unknown")
    
    if hyp_gluc_pred.work_type not in work_type_dict:
        raise HTTPException(status_code=404, detail="Work situation unknown")
    
    if hyp_gluc_pred.residence_type not in residence_type_dict:
        raise HTTPException(status_code=404, detail="Residence type not found")
    
    if hyp_gluc_pred.smoking_status not in smoking_status_dict:
        raise HTTPException(status_code=404, detail="Smoking status not found")


    hyp_gluc_df["gender"] = [hyp_gluc_pred.gender]
    hyp_gluc_df["age"] = [hyp_gluc_pred.age]
    hyp_gluc_df["heart_disease"] = [hyp_gluc_pred.heart_disease]
    hyp_gluc_df["ever_married"]= [hyp_gluc_pred.ever_married]
    hyp_gluc_df["work_type"] = [hyp_gluc_pred.work_type]
    hyp_gluc_df["residence_type"] = [hyp_gluc_pred.residence_type]
    hyp_gluc_df["bmi"] = [hyp_gluc_pred.bmi]
    hyp_gluc_df["smoking_status"] = [hyp_gluc_pred.smoking_status]

    prediction = hypertension_glucose_predictor.predict(hyp_gluc_df)

    if prediction[0][0] == 0 and prediction[0][1] == 0:
        prediction = "You do not have hypertension. Your glucose level is likely to be in the range of: 51-100 mg/dl. "
    elif prediction[0][0] == 0 and prediction[0][1] == 1:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 101-150 mg/dl"
    elif prediction[0][0] == 0 and prediction[0][1] == 2:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 151-200 mg/dl."
    elif prediction[0][0] == 0 and prediction[0][1] == 3:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 201-250 mg/dl."    
    elif prediction[0][0] == 0 and prediction[0][1] == 4:
        prediction = "You do not have hypertension.You are in the danger zone of glucose level (over 250 mg/dl)."   
    elif prediction[0][0] == 1 and prediction[0][1] == 0:
        prediction = "You may have hypertension. Your glucose level is likely to be in the range of: 51-100 mg/dl. "
    elif prediction[0][0] == 1 and prediction[0][1] == 1:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 101-150 mg/dl"
    elif prediction[0][0] == 1 and prediction[0][1] == 2:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 151-200 mg/dl."
    elif prediction[0][0] == 1 and prediction[0][1] == 3:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 201-250 mg/dl."    
    elif prediction[0][0] == 1 and prediction[0][1] == 4:
        prediction = "You may have hypertension.You are in the danger zone of glucose level (over 250 mg/dl)."   
    
    return {'prediction': prediction}

@app.post("/hypertension_glucose_bmi_prediction")
async def create_application(hyp_gluc_bmi_pred: hypertension_glucose_bmi_prediction):

    hyp_gluc_bmi_df = pd.DataFrame()

    if hyp_gluc_bmi_pred.gender not in gender_dict:
        raise HTTPException(status_code=404, detail="Gender not found")

    if hyp_gluc_bmi_pred.heart_disease not in heart_disease_dict:
        raise HTTPException(status_code=404, detail="Heart disease unknown")
    
    if hyp_gluc_bmi_pred.ever_married not in ever_married_dict:
        raise HTTPException(status_code=404, detail="Marital status unknown")
    
    if hyp_gluc_bmi_pred.work_type not in work_type_dict:
        raise HTTPException(status_code=404, detail="Work situation unknown")
    
    if hyp_gluc_bmi_pred.residence_type not in residence_type_dict:
        raise HTTPException(status_code=404, detail="Residence type not found")
    
    if hyp_gluc_bmi_pred.smoking_status not in smoking_status_dict:
        raise HTTPException(status_code=404, detail="Smoking status not found")

    if hyp_gluc_bmi_pred.stroke not in stroke_dict:
        raise HTTPException(status_code=404, detail="have you ever had a stroke? Because you did not answer.")

    hyp_gluc_bmi_df["gender"] = [hyp_gluc_bmi_pred.gender]
    hyp_gluc_bmi_df["age"] = [hyp_gluc_bmi_pred.age]
    hyp_gluc_bmi_df["heart_disease"] = [hyp_gluc_bmi_pred.heart_disease]
    hyp_gluc_bmi_df["ever_married"]= [hyp_gluc_bmi_pred.ever_married]
    hyp_gluc_bmi_df["work_type"] = [hyp_gluc_bmi_pred.work_type]
    hyp_gluc_bmi_df["residence_type"] = [hyp_gluc_bmi_pred.residence_type]
    hyp_gluc_bmi_df["smoking_status"] = [hyp_gluc_bmi_pred.smoking_status]
    hyp_gluc_bmi_df["stroke"] = [hyp_gluc_bmi_pred.stroke]

    prediction = hyp_gluc_bmi_predictor.predict(hyp_gluc_bmi_df)
    
    if prediction[0][0] == 0 and prediction[0][1] == 0 and prediction[0][2]== 0:
        prediction = "You do not have hypertension. Your glucose level is likely to be in the range of: 51-100 mg/dl. Your are under weighted, your bmi is <=17"
    elif prediction[0][0] == 0 and prediction[0][1] == 1 and prediction[0][2]== 0:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 101-150 mg/dl. Your are under weighted, your bmi is <=17"
    elif prediction[0][0] == 0 and prediction[0][1] == 2 and prediction[0][2]== 0:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 151-200 mg/dl.Your are under weighted, your bmi is <=17"
    elif prediction[0][0] == 0 and prediction[0][1] == 3 and prediction[0][2]== 0:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 201-250 mg/dl.Your are under weighted, your bmi is <=17"    
    elif prediction[0][0] == 0 and prediction[0][1] == 4 and prediction[0][2]== 0:
        prediction = "You do not have hypertension.You are in the danger zone of glucose level (over 250 mg/dl).Your are under weighted, your bmi is <=17"   
    elif prediction[0][0] == 1 and prediction[0][1] == 0 and prediction[0][2]== 0:
        prediction = "You may have hypertension. Your glucose level is likely to be in the range of: 51-100 mg/dl. Your are under weighted, your bmi is <=17"
    elif prediction[0][0] == 1 and prediction[0][1] == 1 and prediction[0][2]== 0:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 101-150 mg/dl. Your are under weighted, your bmi is <=17"
    elif prediction[0][0] == 1 and prediction[0][1] == 2 and prediction[0][2]== 0:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 151-200 mg/dl.Your are under weighted, your bmi is <=17"
    elif prediction[0][0] == 1 and prediction[0][1] == 3 and prediction[0][2]== 0:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 201-250 mg/dl.Your are under weighted, your bmi is <=17"    
    elif prediction[0][0] == 1 and prediction[0][1] == 4 and prediction[0][2]== 0:
        prediction = "You may have hypertension.You are in the danger zone of glucose level (over 250 mg/dl).Your are under weighted, your bmi is <=17"   
    elif prediction[0][0] == 0 and prediction[0][1] == 0 and prediction[0][2]== 1:
        prediction = "You do not have hypertension. Your glucose level is likely to be in the range of: 51-100 mg/dl. Your bmi is in the normal range of 18-24"
    elif prediction[0][0] == 0 and prediction[0][1] == 1 and prediction[0][2]== 1:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 101-150 mg/dl. Your bmi is in the normal range of 18-24"
    elif prediction[0][0] == 0 and prediction[0][1] == 2 and prediction[0][2]== 1:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 151-200 mg/dl.Your bmi is in the normal range of 18-24"
    elif prediction[0][0] == 0 and prediction[0][1] == 3 and prediction[0][2]== 1:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 201-250 mg/dl.Your bmi is in the normal range of 18-24"    
    elif prediction[0][0] == 0 and prediction[0][1] == 4 and prediction[0][2]== 1:
        prediction = "You do not have hypertension.You are in the danger zone of glucose level (over 250 mg/dl).Your bmi is in the normal range of 18-24"   
    elif prediction[0][0] == 1 and prediction[0][1] == 0 and prediction[0][2]== 1:
        prediction = "You may have hypertension. Your glucose level is likely to be in the range of: 51-100 mg/dl.Your bmi is in the normal range of 18-24"
    elif prediction[0][0] == 1 and prediction[0][1] == 1 and prediction[0][2]== 1:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 101-150 mg/dl. Your bmi is in the normal range of 18-24"
    elif prediction[0][0] == 1 and prediction[0][1] == 2 and prediction[0][2]== 1:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 151-200 mg/dl.Your bmi is in the normal range of 18-24"
    elif prediction[0][0] == 1 and prediction[0][1] == 3 and prediction[0][2]== 1:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 201-250 mg/dl.Your bmi is in the normal range of 18-24"    
    elif prediction[0][0] == 1 and prediction[0][1] == 4 and prediction[0][2]== 1:
        prediction = "You may have hypertension.You are in the danger zone of glucose level (over 250 mg/dl).Your bmi is in the normal range of 18-24" 
    elif prediction[0][0] == 0 and prediction[0][1] == 0 and prediction[0][2]== 2:
        prediction = "You do not have hypertension. Your glucose level is likely to be in the range of: 51-100 mg/dl. Your bmi is in the range of slightly overweight: 25-30"
    elif prediction[0][0] == 0 and prediction[0][1] == 1 and prediction[0][2]== 2:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 101-150 mg/dl. Your bmi is in the range of slightly overweight: 25-30"
    elif prediction[0][0] == 0 and prediction[0][1] == 2 and prediction[0][2]== 2:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 151-200 mg/dl.Your bmi is in the range of slightly overweight: 25-30"
    elif prediction[0][0] == 0 and prediction[0][1] == 3 and prediction[0][2]== 2:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 201-250 mg/dl.Your bmi is in the range of slightly overweight: 25-30"    
    elif prediction[0][0] == 0 and prediction[0][1] == 4 and prediction[0][2]== 2:
        prediction = "You do not have hypertension.You are in the danger zone of glucose level (over 250 mg/dl).Your bmi is in the range of slightly overweight: 25-30"   
    elif prediction[0][0] == 1 and prediction[0][1] == 0 and prediction[0][2]== 2:
        prediction = "You may have hypertension. Your glucose level is likely to be in the range of: 51-100 mg/dl.Your bmi is in the range of slightly overweight: 25-30"
    elif prediction[0][0] == 1 and prediction[0][1] == 1 and prediction[0][2]== 2:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 101-150 mg/dl. Your bmi is in the range of slightly overweight: 25-30"
    elif prediction[0][0] == 1 and prediction[0][1] == 2 and prediction[0][2]== 2:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 151-200 mg/dl.Your bmi is in the range of slightly overweight: 25-30"
    elif prediction[0][0] == 1 and prediction[0][1] == 3 and prediction[0][2]== 2:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 201-250 mg/dl.Your bmi is in the range of slightly overweight: 25-30"    
    elif prediction[0][0] == 1 and prediction[0][1] == 4 and prediction[0][2]== 2:
        prediction = "You may have hypertension.You are in the danger zone of glucose level (over 250 mg/dl).Your bmi is in the range of slightly overweight: 25-30" 
    elif prediction[0][0] == 0 and prediction[0][1] == 0 and prediction[0][2]== 3:
        prediction = "You do not have hypertension. Your glucose level is likely to be in the range of: 51-100 mg/dl.Your bmi is in the overweight range of 31-35"
    elif prediction[0][0] == 0 and prediction[0][1] == 1 and prediction[0][2]== 3:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 101-150 mg/dl. Your bmi is in the overweight range of 31-35"
    elif prediction[0][0] == 0 and prediction[0][1] == 2 and prediction[0][2]== 3:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 151-200 mg/dl.Your bmi is in the overweight range of 31-35"
    elif prediction[0][0] == 0 and prediction[0][1] == 3 and prediction[0][2]== 3:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 201-250 mg/dl.Your bmi is in the overweight range of 31-35"    
    elif prediction[0][0] == 0 and prediction[0][1] == 4 and prediction[0][2]== 3:
        prediction = "You do not have hypertension.You are in the danger zone of glucose level (over 250 mg/dl).Your bmi is in the overweight range of 31-35"   
    elif prediction[0][0] == 1 and prediction[0][1] == 0 and prediction[0][2]== 3:
        prediction = "You may have hypertension. Your glucose level is likely to be in the range of: 51-100 mg/dl.Your bmi is in the overweight range of 31-35"
    elif prediction[0][0] == 1 and prediction[0][1] == 1 and prediction[0][2]== 3:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 101-150 mg/dl.Your bmi is in the overweight range of 31-35"
    elif prediction[0][0] == 1 and prediction[0][1] == 2 and prediction[0][2]== 3:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 151-200 mg/dl.Your bmi is in the overweight range of 31-35"
    elif prediction[0][0] == 1 and prediction[0][1] == 3 and prediction[0][2]== 3:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 201-250 mg/dl.Your bmi is in the overweight range of 31-35"    
    elif prediction[0][0] == 1 and prediction[0][1] == 4 and prediction[0][2]== 3:
        prediction = "You may have hypertension.You are in the danger zone of glucose level (over 250 mg/dl).Your bmi is in the overweight range of 31-35" 
    elif prediction[0][0] == 0 and prediction[0][1] == 0 and prediction[0][2]== 4:
        prediction = "You do not have hypertension. Your glucose level is likely to be in the range of: 51-100 mg/dl.Your bmi is in the medium overweight range of 35-40"
    elif prediction[0][0] == 0 and prediction[0][1] == 1 and prediction[0][2]==4:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 101-150 mg/dl. Your bmi is in the medium overweight range of 35-40"
    elif prediction[0][0] == 0 and prediction[0][1] == 2 and prediction[0][2]== 4:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 151-200 mg/dl.Your bmi is in the medium overweight range of 35-40"
    elif prediction[0][0] == 0 and prediction[0][1] == 3 and prediction[0][2]== 4:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 201-250 mg/dl.Your bmi is in the medium overweight range of 35-40"    
    elif prediction[0][0] == 0 and prediction[0][1] == 4 and prediction[0][2]== 4:
        prediction = "You do not have hypertension.You are in the danger zone of glucose level (over 250 mg/dl).Your bmi is in the medium overweight range of 35-40"   
    elif prediction[0][0] == 1 and prediction[0][1] == 0 and prediction[0][2]== 4:
        prediction = "You may have hypertension. Your glucose level is likely to be in the range of: 51-100 mg/dl.Your bmi is in the medium overweight range of 35-40"
    elif prediction[0][0] == 1 and prediction[0][1] == 1 and prediction[0][2]== 4:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 101-150 mg/dl.Your bmi is in the medium overweight range of 35-40"
    elif prediction[0][0] == 1 and prediction[0][1] == 2 and prediction[0][2]== 4:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 151-200 mg/dl.Your bmi is in the medium overweight range of 35-40"
    elif prediction[0][0] == 1 and prediction[0][1] == 3 and prediction[0][2]== 4:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 201-250 mg/dl.Your bmi is in the medium overweight range of 35-40"    
    elif prediction[0][0] == 1 and prediction[0][1] == 4 and prediction[0][2]== 4:
        prediction = "You may have hypertension.You are in the danger zone of glucose level (over 250 mg/dl).Your bmi is in the medium overweight range of 35-40" 
    elif prediction[0][0] == 0 and prediction[0][1] == 0 and prediction[0][2]==5:
        prediction = "You do not have hypertension. Your glucose level is likely to be in the range of: 51-100 mg/dl.You have overweight, your bmi is over 41. Please contact your family doctor."
    elif prediction[0][0] == 0 and prediction[0][1] == 1 and prediction[0][2]==5:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 101-150 mg/dl. You have overweight, your bmi is over 41. Please contact your family doctor."
    elif prediction[0][0] == 0 and prediction[0][1] == 2 and prediction[0][2]== 5:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 151-200 mg/dl.You have overweight, your bmi is over 41. Please contact your family doctor."
    elif prediction[0][0] == 0 and prediction[0][1] == 3 and prediction[0][2]== 5:
        prediction = "You do not have hypertension.Your glucose level is likely to be in the range of: 201-250 mg/dl.You have overweight, your bmi is over 41. Please contact your family doctor."    
    elif prediction[0][0] == 0 and prediction[0][1] == 4 and prediction[0][2]== 5:
        prediction = "You do not have hypertension.You are in the danger zone of glucose level (over 250 mg/dl).You have overweight, your bmi is over 41. Please contact your family doctor."   
    elif prediction[0][0] == 1 and prediction[0][1] == 0 and prediction[0][2]== 5:
        prediction = "You may have hypertension. Your glucose level is likely to be in the range of: 51-100 mg/dl.You have overweight, your bmi is over 41. Please contact your family doctor."
    elif prediction[0][0] == 1 and prediction[0][1] == 1 and prediction[0][2]== 5:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 101-150 mg/dl.You have overweight, your bmi is over 41. Please contact your family doctor."
    elif prediction[0][0] == 1 and prediction[0][1] == 2 and prediction[0][2]== 5:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 151-200 mg/dl.You have overweight, your bmi is over 41. Please contact your family doctor."
    elif prediction[0][0] == 1 and prediction[0][1] == 3 and prediction[0][2]== 5:
        prediction = "You may have hypertension.Your glucose level is likely to be in the range of: 201-250 mg/dl.You have overweight, your bmi is over 41. Please contact your family doctor."    
    elif prediction[0][0] == 1 and prediction[0][1] == 4 and prediction[0][2]==5:
        prediction = "You may have hypertension.You are in the danger zone of glucose level (over 250 mg/dl).You have overweight, your bmi is over 41. Please contact your family doctor." 
  
    return {'prediction': prediction}

# uvicorn main:app --reload