# 🚀 Complete Setup Guide for Medical Disease Prediction Project

All 4 Levels: ML Pipeline → Web App → API → Docker

---

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running Level 1 & 2](#running-level-1--2-jupyter-notebook)
4. [Running Level 2a](#running-level-2a-streamlit-web-app)
5. [Running Level 2b](#running-level-2b-shap-explainability)
6. [Running Level 3](#running-level-3-model-optimization)
7. [Running Level 4a](#running-level-4a-fastapi-deployment)
8. [Running Level 4b](#running-level-4b-docker-deployment)
9. [API Usage Examples](#api-usage-examples)
10. [Troubleshooting](#troubleshooting)

---

## 🔧 Prerequisites

### System Requirements
- **Python:** 3.8 or higher
- **OS:** Windows, macOS, or Linux
- **RAM:** 4GB minimum (8GB recommended)
- **Disk:** 500MB for dependencies

### Required Tools
```bash
# Check Python version
python --version  # Should be >= 3.8

# Check pip
pip --version

# Optional: Docker (for Level 4b)
docker --version
```

---

## 💾 Installation

### Step 1: Clone Repository
```bash
cd path/to/your/projects
git clone https://github.com/Palak985/Medical-Disease-Prediction.git
cd Medical-Disease-Prediction
```

### Step 2: Create Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Create Required Directories
```bash
# Windows
mkdir models

# macOS/Linux
mkdir -p models
```

### Step 5: Prepare Dataset
- Ensure `dataset/Diabetes.csv` exists in the project root
- Or download from: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

---

## 📖 Running Level 1 & 2: Jupyter Notebook

### Interactive Learning & Exploration

```bash
# Start Jupyter Lab
jupyter lab

# Or Jupyter Notebook
jupyter notebook
```

Then open: `Project_1_Level_1_Diabetes_Prediction.ipynb`

**Features:**
- ✅ Step-by-step ML pipeline
- ✅ Data exploration
- ✅ Model training
- ✅ Predictions with explanations
- ✅ Interactive visualization

**Run cells in order for best results**

---

## 🎨 Running Level 2a: Streamlit Web App

### Interactive UI for Predictions

```bash
streamlit run streamlit_app.py
```

**Access:** http://localhost:8501

**Features:**
- 🎯 Single Patient Prediction
- 📊 Model Performance Dashboard
- 📈 Data Exploration Tools
- 📥 Batch CSV Upload
- 📉 Multiple Model Comparison

### Features Overview:

#### 1. Make Prediction
- **Manual Entry:** Input patient health metrics
- **CSV Upload:** Batch predictions for multiple patients
- **Model Selection:** Choose Logistic Regression, Decision Tree, or Random Forest

#### 2. Model Performance
- Accuracy comparison across models
- Confusion matrices
- Classification reports
- ROC curves

#### 3. Data Exploration
- Feature distributions
- Correlation matrices
- Statistical summaries
- Diabetes prevalence analysis

#### 4. About
- Project information
- Dataset details
- Use cases and applications

### Example Patient Input:
```
Pregnancies: 2
Glucose: 120 mg/dL
Blood Pressure: 70 mmHg
Skin Thickness: 25 mm
Insulin: 100 mIU/L
BMI: 28.5 kg/m²
Diabetes Pedigree: 0.5
Age: 35 years
```

---

## 🔍 Running Level 2b: SHAP Explainability

### Model Interpretation & Explanation

```bash
streamlit run shap_explainability.py
```

**Access:** http://localhost:8501

**Features:**
- 🎯 Global Feature Importance
- 📊 SHAP Summary Plots
- 👤 Individual Prediction Explanation
- 💡 Feature Impact Analysis

### How It Works:

#### 1. Feature Importance
- See which features matter most globally
- Understand feature contributions

#### 2. Summary Plots
- **Bar Plot:** Overall importance ranking
- **Violin Plot:** SHAP value distributions
- **Dependence Plot:** Feature value vs impact

#### 3. Individual Explanation
- Explain specific patient predictions
- Force plots showing feature contributions
- Waterfall plots for cumulative effects

#### 4. Feature Impact
- How specific features affect predictions
- Scatter plots with SHAP values
- Patient outcome visualization

### Example Analysis:
```
For Patient ID #42:
- Glucose: +0.25 SHAP (increases diabetes risk)
- BMI: +0.18 SHAP (increases diabetes risk)
- Age: -0.05 SHAP (decreases diabetes risk)
- Prediction: Diabetes (75% confidence)
```

---

## 🚀 Running Level 3: Model Optimization

### Hyperparameter Tuning & Model Comparison

```bash
python level_3_model_optimization.py
```

**What It Does:**
- 🔧 Hyperparameter tuning for 5 models
- 📊 Cross-validation analysis
- 📈 Performance metrics comparison
- 💾 Saves best models to `/models` folder
- 📉 Generates comparison visualizations

**Models Tested:**
1. Logistic Regression
2. Decision Tree
3. Random Forest
4. Gradient Boosting
5. XGBoost

**Output Files:**
- `model_comparison.png` - Performance comparison charts
- `cv_comparison.png` - Cross-validation scores
- `best_model_metrics.png` - Best model metrics
- `feature_importance.png` - Feature importance ranking
- `models/` - Saved tuned models (.pkl files)

**Expected Results:**
```
Logistic Regression: 77% accuracy
Decision Tree:       73% accuracy
Random Forest:       82% accuracy ⭐ BEST
Gradient Boosting:   80% accuracy
XGBoost:            81% accuracy
```

### Running Time:
- **Total:** ~10-15 minutes (depending on CPU)
- Models are parallelized for faster execution

---

## 🔌 Running Level 4a: FastAPI Deployment

### Production-Ready REST API

```bash
python level_4_api.py
```

**Access:**
- 🔗 API: http://localhost:8000
- 📚 Docs: http://localhost:8000/docs (Interactive Swagger UI)
- 📖 ReDoc: http://localhost:8000/redoc (Alternative docs)

### Quick Test in Browser:
1. Go to: http://localhost:8000/docs
2. Click on `/predict` endpoint
3. Click "Try it out"
4. Enter sample patient data
5. Click "Execute"

### API Endpoints:

#### 1. Single Prediction
```bash
POST /predict
```

**Request:**
```json
{
  "pregnancies": 2,
  "glucose": 120,
  "blood_pressure": 70,
  "skin_thickness": 25,
  "insulin": 100,
  "bmi": 28.5,
  "dpf": 0.5,
  "age": 35
}
```

**Response:**
```json
{
  "prediction": "No Diabetes",
  "probability": 0.25,
  "confidence": "High",
  "risk_level": "Low",
  "recommendation": "✅ No immediate action needed. Maintain healthy lifestyle."
}
```

#### 2. Batch Prediction
```bash
POST /predict-batch
```

**Request:**
```json
{
  "patients": [
    {
      "pregnancies": 2,
      "glucose": 120,
      ...
    },
    {
      "pregnancies": 1,
      "glucose": 150,
      ...
    }
  ]
}
```

#### 3. Health Check
```bash
GET /health
```

#### 4. Get Features
```bash
GET /features
```

### Using cURL:
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 2,
    "glucose": 120,
    "blood_pressure": 70,
    "skin_thickness": 25,
    "insulin": 100,
    "bmi": 28.5,
    "dpf": 0.5,
    "age": 35
  }'

# Health check
curl http://localhost:8000/health

# Get API docs
curl http://localhost:8000/api-docs-json
```

### Using Python:
```python
import requests

# Single prediction
url = "http://localhost:8000/predict"
patient = {
    "pregnancies": 2,
    "glucose": 120,
    "blood_pressure": 70,
    "skin_thickness": 25,
    "insulin": 100,
    "bmi": 28.5,
    "dpf": 0.5,
    "age": 35
}

response = requests.post(url, json=patient)
result = response.json()

print(f"Prediction: {result['prediction']}")
print(f"Probability: {result['probability']:.1%}")
print(f"Recommendation: {result['recommendation']}")
```

### Using JavaScript/Node.js:
```javascript
const patient = {
  pregnancies: 2,
  glucose: 120,
  blood_pressure: 70,
  skin_thickness: 25,
  insulin: 100,
  bmi: 28.5,
  dpf: 0.5,
  age: 35
};

fetch('http://localhost:8000/predict', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify(patient)
})
.then(res => res.json())
.then(data => console.log(data));
```

---

## 🐳 Running Level 4b: Docker Deployment

### Containerized Application

### Option 1: Docker Compose (Easiest)
```bash
# Start all services
docker-compose up --build

# Or in background
docker-compose up -d --build

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

**Access:**
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Streamlit: http://localhost:8501 (if enabled)

### Option 2: Manual Docker
```bash
# Build image
docker build -t medical-prediction:latest .

# Run container
docker run -d \
  --name medical-prediction-container \
  -p 8000:8000 \
  -v $(pwd)/dataset:/app/dataset:ro \
  -v $(pwd)/models:/app/models:ro \
  medical-prediction:latest

# View logs
docker logs -f medical-prediction-container

# Stop container
docker stop medical-prediction-container
docker rm medical-prediction-container
```

### Docker Commands Reference:
```bash
# Build with specific version
docker build -t medical-prediction:1.0 .

# Push to Docker Hub (after login)
docker login
docker tag medical-prediction:latest your-username/medical-prediction:latest
docker push your-username/medical-prediction:latest

# Pull from Docker Hub
docker pull your-username/medical-prediction:latest

# Run with custom port
docker run -p 9000:8000 medical-prediction:latest

# Check running containers
docker ps

# Check image size
docker images
```

### Health Check:
```bash
# Using curl
curl http://localhost:8000/health

# Using Docker
docker exec medical-prediction-container curl http://localhost:8000/health
```

---

## 📚 API Usage Examples

### Python Example
```python
import requests
import pandas as pd

# API URL
BASE_URL = "http://localhost:8000"

# Example 1: Single prediction
def predict_single_patient():
    patient = {
        "pregnancies": 2,
        "glucose": 150,
        "blood_pressure": 75,
        "skin_thickness": 30,
        "insulin": 120,
        "bmi": 31.0,
        "dpf": 0.6,
        "age": 45
    }
    
    response = requests.post(f"{BASE_URL}/predict", json=patient)
    return response.json()

# Example 2: Batch prediction from CSV
def predict_batch_from_csv(csv_file):
    # Read CSV
    df = pd.read_csv(csv_file)
    
    # Convert to list of dicts
    patients = df.to_dict('records')
    
    # Create request
    request_data = {"patients": patients}
    
    # Make prediction
    response = requests.post(f"{BASE_URL}/predict-batch", json=request_data)
    
    # Save results
    results_df = pd.DataFrame(response.json()['predictions'])
    results_df.to_csv('predictions.csv', index=False)
    
    return response.json()

# Run examples
if __name__ == "__main__":
    print("Single Prediction:")
    result = predict_single_patient()
    print(result)
    
    print("\nBatch Prediction:")
    batch_result = predict_batch_from_csv('patients.csv')
    print(batch_result['summary'])
```

### JavaScript/Frontend Example
```javascript
const API_URL = 'http://localhost:8000';

async function predictDiabetes() {
  const patient = {
    pregnancies: 2,
    glucose: parseInt(document.getElementById('glucose').value),
    blood_pressure: parseInt(document.getElementById('bp').value),
    skin_thickness: parseInt(document.getElementById('skin').value),
    insulin: parseInt(document.getElementById('insulin').value),
    bmi: parseFloat(document.getElementById('bmi').value),
    dpf: parseFloat(document.getElementById('dpf').value),
    age: parseInt(document.getElementById('age').value)
  };
  
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(patient)
    });
    
    const result = await response.json();
    
    // Display results
    document.getElementById('prediction').textContent = result.prediction;
    document.getElementById('probability').textContent = 
      `${(result.probability * 100).toFixed(1)}%`;
    document.getElementById('recommendation').textContent = 
      result.recommendation;
      
  } catch (error) {
    console.error('Error:', error);
  }
}
```

---

## 🐛 Troubleshooting

### Issue: "Module not found" error

**Solution:**
```bash
# Verify virtual environment is activated
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Issue: Port 8000 or 8501 already in use

**Solution:**
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# macOS/Linux
lsof -i :8000
kill -9 <PID>

# Or use different port
streamlit run streamlit_app.py --server.port 8502
uvicorn level_4_api:app --port 8001
```

### Issue: Models not found

**Solution:**
```bash
# Run Level 3 first to generate models
python level_3_model_optimization.py

# Check models folder
ls models/  # macOS/Linux
dir models  # Windows
```

### Issue: Docker build fails

**Solution:**
```bash
# Clean build
docker build --no-cache -t medical-prediction:latest .

# Check Docker daemon
docker ps

# Rebuild from scratch
docker system prune -a
docker build -t medical-prediction:latest .
```

### Issue: SHAP plots not rendering

**Solution:**
```bash
# Reinstall SHAP
pip uninstall shap -y
pip install shap==0.43.0

# Or use alternative installation
pip install -U shap
```

### Issue: Streamlit slow/freezing

**Solution:**
```bash
# Clear cache
streamlit cache clear

# Run with higher verbosity
streamlit run streamlit_app.py --logger.level=debug

# Check available memory
# Add this to code:
import psutil
print(f"Memory: {psutil.virtual_memory().percent}%")
```

---

## 📊 Performance Benchmarks

| Task | Time | Machine |
|------|------|---------|
| Level 1 (Notebook) | 5 min | i5, 8GB RAM |
| Level 2a (Streamlit) | Instant | i5, 8GB RAM |
| Level 2b (SHAP) | 2-3 min | i5, 8GB RAM |
| Level 3 (Optimization) | 10-15 min | i5, 8GB RAM |
| Level 4a (API) | Instant per request | i5, 8GB RAM |
| Level 4b (Docker) | 2-3 min to build | i5, 8GB RAM |

---

## 🎯 What's Next?

After completing all 4 levels:

1. **Deploy to Cloud:**
   - AWS, GCP, Azure, Heroku
   - Docker + Kubernetes

2. **Add Advanced Features:**
   - User authentication
   - Database integration
   - Real-time monitoring
   - A/B testing

3. **Expand Project:**
   - Other diseases (heart, cancer, etc.)
   - Multi-class classification
   - Time-series predictions
   - Integration with EHR systems

---

## 📞 Support

**Got stuck?**
- Check [Troubleshooting](#troubleshooting) section
- Read error messages carefully
- Check GitHub Issues
- Review documentation links

**Want to contribute?**
- Fork the repository
- Create feature branch
- Make improvements
- Submit Pull Request

---

## 📄 License

MIT License - See LICENSE file

---

**Happy coding! 🚀**