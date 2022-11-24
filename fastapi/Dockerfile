FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR /fastapi

RUN pip install sklearn joblib pandas lightgbm imblearn xgboost pydantic

#
COPY . ./

#
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]