from pydantic import BaseModel


class stroke_prediction(BaseModel):
    gender: str
    age: float
    hypertension: str
    heart_disease: str
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

class hypertension_prediction(BaseModel):
    gender: str
    age: float
    heart_disease: str
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    bmi: float
    smoking_status: str

class glucose_prediction(BaseModel):
    gender: str
    age: float
    hypertension: str
    heart_disease: str
    ever_married: str
    work_type: str
    residence_type: str
    bmi: float
    smoking_status: str
    heart_hypertenz: str

class bmi_prediction(BaseModel):
    gender: str
    age: float
    hypertension: str
    heart_disease: str
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    smoking_status: str
    heart_hypertenz: str

class hypertension_glucose_prediction(BaseModel):
    gender: str
    age: float
    heart_disease: str
    ever_married: str
    work_type: str
    residence_type: str
    bmi: float
    smoking_status: str

class hypertension_bmi_prediction(BaseModel):
    gender: str
    age: float
    heart_disease: str
    ever_married: str
    work_type: str
    residence_type: str
    avg_glucose_level: float
    smoking_status: str

class glucose_bmi_prediction(BaseModel):
    gender: str
    age: float
    hypertension: str
    heart_disease: str
    ever_married: str
    work_type: str
    residence_type: str
    smoking_status: str
    heart_hypertenz: str

class hypertension_glucose_bmi_prediction(BaseModel):
    gender: str
    age: float
    heart_disease: str
    ever_married: str
    work_type: str
    residence_type: str
    smoking_status: str
    stroke: str


gender_dict = {
  "male": "male",
  "female": "female",
  }

heart_disease_dict = {
  "yes": "yes",
  "no": "no",
}

hypertension_dict = {
  "yes": "yes",
  "no": "no",
}

ever_married_dict = {
  "yes": "yes",
  "no": "no",
}

work_type_dict = {
  "private":"private",
  "self-employed":"self-employed",
  "not_working": "not_working",
  "govt_job" : "govt_job",
}

residence_type_dict = {
    "urban": "urban",
    "rural": "rural",
}

smoking_status_dict = {
    "never smoked" : "never smoked", 
    "smokes" : "smokes",
    "unknown": "unknown"
}

heart_hypertenz_dict = {
  "yes": "yes",
  "no": "no",
}

stroke_dict = {
  "yes": "yes",
  "no": "no",
}