"""
Level 4a: FastAPI Medical Disease Prediction API
Production-ready REST API for diabetes risk prediction
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import uvicorn
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ============= APP INITIALIZATION =============
app = FastAPI(
    title="Medical Disease Prediction API",
    description="AI-powered REST API for diabetes risk prediction",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= LOAD MODELS =============
@app.on_event("startup")
async def load_models():
    """Load trained models and scaler on startup"""
    global model, scaler
    try:
        # Load model (try best tuned model first, fallback to original)
        try:
            model = pickle.load(open('models/random_forest_tuned.pkl', 'rb'))
            logger.info("✓ Loaded tuned Random Forest model")
        except:
            model = pickle.load(open('models/random_forest_model.pkl', 'rb'))
            logger.info("✓ Loaded baseline Random Forest model")
        
        scaler = pickle.load(open('models/scaler.pkl', 'rb'))
        logger.info("✓ Loaded feature scaler")
        logger.info("🚀 Models loaded successfully!")
    except Exception as e:
        logger.error(f"❌ Failed to load models: {e}")
        raise

# ============= DATA MODELS =============
class PatientData(BaseModel):
    """Single patient health metrics"""
    pregnancies: int = Field(..., ge=0, le=20, description="Number of pregnancies")
    glucose: int = Field(..., ge=0, le=300, description="Plasma glucose concentration (mg/dL)")
    blood_pressure: int = Field(..., ge=0, le=200, description="Diastolic blood pressure (mmHg)")
    skin_thickness: int = Field(..., ge=0, le=100, description="Triceps skin fold thickness (mm)")
    insulin: int = Field(..., ge=0, le=1000, description="2-Hour serum insulin (mIU/L)")
    bmi: float = Field(..., ge=0, le=60, description="Body Mass Index (kg/m²)")
    dpf: float = Field(..., ge=0, le=2.4, description="Diabetes Pedigree Function")
    age: int = Field(..., ge=1, le=120, description="Age in years")
    
    class Config:
        example = {
            "pregnancies": 2,
            "glucose": 120,
            "blood_pressure": 70,
            "skin_thickness": 25,
            "insulin": 100,
            "bmi": 28.5,
            "dpf": 0.5,
            "age": 35
        }

class PredictionResponse(BaseModel):
    """Single prediction response"""
    prediction: str = Field(..., description="Prediction: 'Diabetes' or 'No Diabetes'")
    probability: float = Field(..., ge=0, le=1, description="Probability of diabetes (0-1)")
    confidence: str = Field(..., description="Confidence level: 'High', 'Medium', or 'Low'")
    risk_level: str = Field(..., description="Risk categorization")
    recommendation: str = Field(..., description="Medical recommendation")

class BatchPredictionRequest(BaseModel):
    """Batch prediction request"""
    patients: List[PatientData] = Field(..., min_items=1, max_items=1000, description="List of patients")

class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    total_patients: int
    predictions: List[PredictionResponse]
    summary: dict = Field(..., description="Summary statistics")

class HealthStatus(BaseModel):
    """API health status"""
    status: str
    timestamp: str
    model_info: dict
    version: str

# ============= UTILITY FUNCTIONS =============
def analyze_prediction(probability):
    """Determine confidence and recommendation based on probability"""
    if probability < 0.3:
        confidence = "High"
        risk_level = "Low"
        recommendation = "✅ No immediate action needed. Maintain healthy lifestyle."
    elif probability < 0.6:
        confidence = "Medium"
        risk_level = "Moderate"
        recommendation = "⚠️ Monitor health metrics regularly. Consider lifestyle modifications."
    else:
        confidence = "High"
        risk_level = "High"
        recommendation = "🔴 Consult healthcare professional immediately for detailed assessment."
    
    return confidence, risk_level, recommendation

def prepare_patient_data(patient: PatientData) -> np.ndarray:
    """Convert patient data to model input format"""
    data = np.array([[
        patient.pregnancies,
        patient.glucose,
        patient.blood_pressure,
        patient.skin_thickness,
        patient.insulin,
        patient.bmi,
        patient.dpf,
        patient.age
    ]])
    return scaler.transform(data)

# ============= ENDPOINTS =============

@app.get("/", tags=["Info"])
async def root():
    """Root endpoint - API information"""
    return {
        "name": "Medical Disease Prediction API",
        "version": "1.0.0",
        "description": "AI-powered REST API for diabetes risk prediction",
        "docs": "/docs",
        "redoc": "/redoc"
    }

@app.get("/health", response_model=HealthStatus, tags=["Info"])
async def health_check():
    """Check API health and model status"""
    try:
        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "model_info": {
                "type": "Random Forest Classifier",
                "n_features": 8,
                "features": ["pregnancies", "glucose", "blood_pressure", "skin_thickness", 
                            "insulin", "bmi", "dpf", "age"]
            },
            "version": "1.0.0"
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {"status": "error", "message": str(e), "timestamp": datetime.now().isoformat()}

@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(patient: PatientData):
    """
    Single patient prediction
    
    Returns diabetes risk prediction with probability and recommendations
    """
    try:
        logger.info(f"Processing single prediction for patient age {patient.age}")
        
        # Prepare data
        X_scaled = prepare_patient_data(patient)
        
        # Make prediction
        prediction = model.predict(X_scaled)[0]
        probability = model.predict_proba(X_scaled)[0][1]
        
        # Analyze prediction
        confidence, risk_level, recommendation = analyze_prediction(probability)
        
        response = {
            "prediction": "Diabetes" if prediction == 1 else "No Diabetes",
            "probability": float(probability),
            "confidence": confidence,
            "risk_level": risk_level,
            "recommendation": recommendation
        }
        
        logger.info(f"✓ Prediction complete: {response['prediction']} ({probability:.1%})")
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Batch prediction for multiple patients
    
    Accepts up to 1000 patients and returns predictions for all
    """
    try:
        logger.info(f"Processing batch prediction for {len(request.patients)} patients")
        
        predictions_list = []
        
        for patient in request.patients:
            # Prepare data
            X_scaled = prepare_patient_data(patient)
            
            # Make prediction
            prediction = model.predict(X_scaled)[0]
            probability = model.predict_proba(X_scaled)[0][1]
            
            # Analyze prediction
            confidence, risk_level, recommendation = analyze_prediction(probability)
            
            predictions_list.append({
                "prediction": "Diabetes" if prediction == 1 else "No Diabetes",
                "probability": float(probability),
                "confidence": confidence,
                "risk_level": risk_level,
                "recommendation": recommendation
            })
        
        # Calculate summary
        diabetes_count = sum(1 for p in predictions_list if p['prediction'] == 'Diabetes')
        high_risk_count = sum(1 for p in predictions_list if p['risk_level'] == 'High')
        avg_probability = np.mean([p['probability'] for p in predictions_list])
        
        summary = {
            "total_processed": len(request.patients),
            "positive_cases": diabetes_count,
            "high_risk_cases": high_risk_count,
            "average_probability": float(avg_probability),
            "positive_rate": float(diabetes_count / len(request.patients))
        }
        
        logger.info(f"✓ Batch complete: {diabetes_count} positive cases out of {len(request.patients)}")
        
        return {
            "total_patients": len(request.patients),
            "predictions": predictions_list,
            "summary": summary
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/features", tags=["Info"])
async def get_features():
    """Get list of required features with descriptions"""
    return {
        "features": [
            {
                "name": "pregnancies",
                "type": "integer",
                "range": "0-20",
                "unit": "count",
                "description": "Number of times pregnant"
            },
            {
                "name": "glucose",
                "type": "integer",
                "range": "0-300",
                "unit": "mg/dL",
                "description": "Plasma glucose concentration"
            },
            {
                "name": "blood_pressure",
                "type": "integer",
                "range": "0-200",
                "unit": "mmHg",
                "description": "Diastolic blood pressure"
            },
            {
                "name": "skin_thickness",
                "type": "integer",
                "range": "0-100",
                "unit": "mm",
                "description": "Triceps skin fold thickness"
            },
            {
                "name": "insulin",
                "type": "integer",
                "range": "0-1000",
                "unit": "mIU/L",
                "description": "2-Hour serum insulin level"
            },
            {
                "name": "bmi",
                "type": "float",
                "range": "0-60",
                "unit": "kg/m²",
                "description": "Body Mass Index"
            },
            {
                "name": "dpf",
                "type": "float",
                "range": "0-2.4",
                "unit": "score",
                "description": "Diabetes Pedigree Function"
            },
            {
                "name": "age",
                "type": "integer",
                "range": "1-120",
                "unit": "years",
                "description": "Patient age"
            }
        ]
    }

@app.get("/api-docs-json", tags=["Info"])
async def api_documentation():
    """Get API documentation in JSON format"""
    return {
        "API": "Medical Disease Prediction",
        "Version": "1.0.0",
        "Endpoints": {
            "/predict": {
                "method": "POST",
                "description": "Single patient prediction",
                "example": "/docs"
            },
            "/predict-batch": {
                "method": "POST",
                "description": "Batch predictions for multiple patients",
                "max_patients": 1000
            },
            "/health": {
                "method": "GET",
                "description": "API health status"
            },
            "/features": {
                "method": "GET",
                "description": "Get required features"
            }
        },
        "Authentication": "None (Public API)",
        "Rate Limit": "No limit (add in production!"
    }

# ============= ERROR HANDLERS =============
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return HTTPException(status_code=400, detail=str(exc))

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return HTTPException(status_code=500, detail="Internal server error")

# ============= MAIN =============
if __name__ == "__main__":
    print("="*70)
    print("🚀 Starting Medical Disease Prediction API")
    print("="*70)
    print("\n📍 Server running at: http://127.0.0.1:8000")
    print("📚 API Docs at:      http://127.0.0.1:8000/docs")
    print("🔄 ReDoc at:         http://127.0.0.1:8000/redoc")
    print("\n" + "="*70)
    
    uvicorn.run(
        "level_4_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )