from fastapi import FastAPI, Query
from inference_onnx import ColaONNXPredictor

app = FastAPI(title="MLOps Basics App")

predictor = ColaONNXPredictor("./models/model.onnx")

@app.get("/")
async def home_page():
    return "<h2>This is a sample NLP Project</h2>"

@app.get("/predict")
async def get_prediction(text: str = Query(..., description="Text to classify")):
    result = predictor.predict(text)
    return result