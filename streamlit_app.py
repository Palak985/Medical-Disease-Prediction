"""
Streamlit Web App for Medical Disease Prediction
Interactive UI for diabetes risk prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Diabetes Risk Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        font-size: 3em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
    }
    .risk-high {
        background-color: #ffcccc;
        padding: 15px;
        border-radius: 8px;
        color: #cc0000;
    }
    .risk-low {
        background-color: #ccffcc;
        padding: 15px;
        border-radius: 8px;
        color: #009900;
    }
    </style>
""", unsafe_allow_html=True)

# Load and prepare data
@st.cache_resource
def load_data():
    df = pd.read_csv('dataset/Diabetes.csv')
    # Handle missing values
    zero_cols = ['Diastolic blood pressure', 'Triceps skin fold thickness', 
                 '2-Hour serum insulin', 'Body mass index']
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    df[zero_cols] = df[zero_cols].fillna(df[zero_cols].median())
    return df

@st.cache_resource
def train_models(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=42),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        trained_models[name] = {
            'model': model,
            'scaler': scaler,
            'X_test': X_test_scaled,
            'y_test': y_test,
            'y_pred': model.predict(X_test_scaled),
            'accuracy': accuracy_score(y_test, model.predict(X_test_scaled))
        }
    
    return trained_models, X_train, X_test, y_train, y_test, scaler

# Header
st.markdown('<div class="main-header">🏥 Diabetes Risk Prediction System</div>', unsafe_allow_html=True)
st.markdown("**AI-powered system for predicting diabetes risk from patient health metrics**")
st.divider()

# Load data
df = load_data()
X = df.drop('Class variable', axis=1)
y = (df['Class variable'] == 'YES').astype(int)

# Train models
models_dict, X_train, X_test, y_train, y_test, scaler = train_models(X, y)

# Sidebar for navigation
st.sidebar.title("📋 Navigation")
page = st.sidebar.radio(
    "Select a page:",
    ["🎯 Make Prediction", "📊 Model Performance", "📈 Data Exploration", "ℹ️ About"]
)

# ============= PAGE 1: MAKE PREDICTION =============
if page == "🎯 Make Prediction":
    st.header("Make a Prediction")
    st.write("Enter patient health metrics to predict diabetes risk")
    
    # Model selection
    col1, col2 = st.columns([1, 1])
    with col1:
        selected_model = st.selectbox(
            "Select Model:",
            list(models_dict.keys()),
            help="Choose which model to use for prediction"
        )
    
    with col2:
        st.metric("Model Accuracy", f"{models_dict[selected_model]['accuracy']:.1%}")
    
    st.divider()
    
    # Input method selection
    input_method = st.radio("Input Method:", ["Manual Entry", "Upload CSV"])
    
    if input_method == "Manual Entry":
        st.subheader("Enter Patient Data")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
            glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=300, value=120)
        
        with col2:
            blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=70)
            skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=100, value=25)
        
        with col3:
            insulin = st.number_input("Insulin (mIU/L)", min_value=0, max_value=1000, value=100)
            bmi = st.number_input("BMI (kg/m²)", min_value=0.0, max_value=60.0, value=28.5)
        
        with col4:
            dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.4, value=0.5)
            age = st.number_input("Age (years)", min_value=1, max_value=120, value=35)
        
        # Prepare input data
        patient_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, 
                                 insulin, bmi, dpf, age]])
        
        # Make prediction
        if st.button("🔮 Predict", use_container_width=True, key="predict_button"):
            patient_scaled = scaler.transform(patient_data)
            model = models_dict[selected_model]['model']
            
            prediction = model.predict(patient_scaled)[0]
            probability = model.predict_proba(patient_scaled)[0]
            
            st.divider()
            
            # Display results
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.subheader("📋 Patient Summary")
                summary_df = pd.DataFrame({
                    'Metric': ['Pregnancies', 'Glucose', 'Blood Pressure', 'BMI', 'Age', 'Diabetes Pedigree'],
                    'Value': [pregnancies, f'{glucose} mg/dL', f'{blood_pressure} mmHg', 
                             f'{bmi:.1f}', f'{age} yrs', f'{dpf:.2f}']
                })
                st.dataframe(summary_df, use_container_width=True, hide_index=True)
            
            with col2:
                st.subheader("🎯 Prediction Result")
                
                if prediction == 0:
                    st.markdown(
                        f'<div class="risk-low"><h2>✅ LOW RISK</h2><p>No Diabetes Detected</p><p style="font-size:1.2em">Confidence: {probability[0]:.1%}</p></div>',
                        unsafe_allow_html=True
                    )
                else:
                    st.markdown(
                        f'<div class="risk-high"><h2>⚠️ HIGH RISK</h2><p>Diabetes Detected</p><p style="font-size:1.2em">Confidence: {probability[1]:.1%}</p></div>',
                        unsafe_allow_html=True
                    )
                
                st.metric("Diabetes Probability", f"{probability[1]:.1%}")
    
    else:  # CSV Upload
        st.subheader("Upload Patient Data (CSV)")
        uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
        
        if uploaded_file is not None:
            try:
                uploaded_df = pd.read_csv(uploaded_file)
                st.write("Preview:")
                st.dataframe(uploaded_df.head(), use_container_width=True)
                
                if st.button("🔮 Predict All", use_container_width=True):
                    # Make predictions
                    uploaded_scaled = scaler.transform(uploaded_df)
                    model = models_dict[selected_model]['model']
                    
                    predictions = model.predict(uploaded_scaled)
                    probabilities = model.predict_proba(uploaded_scaled)
                    
                    # Create results dataframe
                    results_df = uploaded_df.copy()
                    results_df['Prediction'] = ['Diabetes' if p == 1 else 'No Diabetes' for p in predictions]
                    results_df['Diabetes_Probability'] = probabilities[:, 1]
                    results_df['Risk_Level'] = results_df['Diabetes_Probability'].apply(
                        lambda x: 'High' if x > 0.6 else ('Medium' if x > 0.4 else 'Low')
                    )
                    
                    st.write("### Predictions:")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Download results
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name="diabetes_predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error: {e}")

# ============= PAGE 2: MODEL PERFORMANCE =============
elif page == "📊 Model Performance":
    st.header("Model Performance Comparison")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Patients (Test)", len(y_test))
    with col2:
        st.metric("Diabetes Cases", (y_test == 1).sum())
    with col3:
        st.metric("Healthy Cases", (y_test == 0).sum())
    
    st.divider()
    
    # Model comparison
    st.subheader("Model Accuracy Comparison")
    accuracies = {name: data['accuracy'] for name, data in models_dict.items()}
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    bars = ax.bar(accuracies.keys(), accuracies.values(), color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1%}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    st.pyplot(fig)
    
    st.divider()
    
    # Detailed model metrics
    st.subheader("Detailed Metrics by Model")
    
    for model_name, model_data in models_dict.items():
        with st.expander(f"📊 {model_name} Details"):
            col1, col2, col3, col4 = st.columns(4)
            
            y_pred = model_data['y_pred']
            cm = confusion_matrix(model_data['y_test'], y_pred)
            
            with col1:
                st.metric("Accuracy", f"{model_data['accuracy']:.1%}")
            
            with col2:
                tn, fp, fn, tp = cm.ravel()
                specificity = tn / (tn + fp)
                st.metric("Specificity", f"{specificity:.1%}")
            
            with col3:
                sensitivity = tp / (tp + fn)
                st.metric("Sensitivity", f"{sensitivity:.1%}")
            
            with col4:
                if (tp + fp) > 0:
                    precision = tp / (tp + fp)
                    st.metric("Precision", f"{precision:.1%}")
            
            # Confusion Matrix
            fig, ax = plt.subplots(figsize=(8, 6))
            cm_display = confusion_matrix(model_data['y_test'], y_pred)
            sns.heatmap(cm_display, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No Diabetes', 'Diabetes'],
                       yticklabels=['No Diabetes', 'Diabetes'])
            ax.set_title(f'Confusion Matrix - {model_name}')
            ax.set_ylabel('Actual')
            ax.set_xlabel('Predicted')
            st.pyplot(fig)
            
            # Classification Report
            report = classification_report(model_data['y_test'], y_pred, 
                                         target_names=['No Diabetes', 'Diabetes'],
                                         output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df, use_container_width=True)

# ============= PAGE 3: DATA EXPLORATION =============
elif page == "📈 Data Exploration":
    st.header("Data Exploration & Analysis")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Patients", len(df))
    with col2:
        st.metric("Features", len(X.columns))
    with col3:
        diabetes_rate = (df['Class variable'] == 'YES').mean()
        st.metric("Diabetes Rate", f"{diabetes_rate:.1%}")
    
    st.divider()
    
    # Feature distributions
    st.subheader("Feature Distributions")
    
    features = X.columns.tolist()
    selected_features = st.multiselect(
        "Select features to visualize:",
        features,
        default=features[:4]
    )
    
    if selected_features:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for idx, feat in enumerate(selected_features[:4]):
            ax = axes[idx]
            df[df['Class variable'] == 'NO'][feat].hist(ax=ax, alpha=0.6, label='No Diabetes', bins=20, color='steelblue')
            df[df['Class variable'] == 'YES'][feat].hist(ax=ax, alpha=0.6, label='Diabetes', bins=20, color='salmon')
            ax.set_title(feat, fontsize=12, fontweight='bold')
            ax.set_xlabel('Value')
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(alpha=0.3)
        
        st.pyplot(fig)
    
    st.divider()
    
    # Correlation
    st.subheader("Feature Correlations")
    
    corr_matrix = X.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
    ax.set_title('Feature Correlation Matrix')
    st.pyplot(fig)
    
    st.divider()
    
    # Statistical summary
    st.subheader("Statistical Summary")
    st.dataframe(df.describe(), use_container_width=True)

# ============= PAGE 4: ABOUT =============
elif page == "ℹ️ About":
    st.header("About This Project")
    
    st.markdown("""
    ### 🏥 Medical Disease Prediction System
    
    This interactive web application predicts diabetes risk using machine learning models trained on the 
    **Pima Indians Diabetes Database**.
    
    #### 📊 Dataset Information
    - **Records:** 768 patient cases
    - **Features:** 8 medical/health metrics
    - **Target:** Binary (Diabetes: YES/NO)
    - **Source:** UCI Machine Learning Repository
    
    #### 🤖 Models Used
    1. **Logistic Regression** - Fast, interpretable baseline
    2. **Decision Tree** - Easy to understand predictions
    3. **Random Forest** - Ensemble learning, best performance
    
    #### 🔍 Patient Features
    | Feature | Unit |
    |---------|------|
    | Pregnancies | Count |
    | Glucose | mg/dL |
    | Blood Pressure | mmHg |
    | Skin Thickness | mm |
    | Insulin | mIU/L |
    | BMI | kg/m² |
    | Diabetes Pedigree Function | Score |
    | Age | Years |
    
    #### 🎯 Use Cases
    - ✅ Early diabetes detection
    - ✅ Patient risk stratification
    - ✅ Clinical decision support
    - ✅ Healthcare research
    
    #### 💡 Key Features
    - Interactive single patient prediction
    - Batch prediction (CSV upload)
    - Real-time model performance metrics
    - Data exploration tools
    - Multiple ML algorithms
    
    #### 📚 Technologies
    - **Python** - Core programming
    - **Streamlit** - Web interface
    - **Scikit-learn** - ML algorithms
    - **Pandas** - Data manipulation
    - **Matplotlib & Seaborn** - Visualization
    
    #### ⚠️ Disclaimer
    This tool is for **educational and research purposes only**. 
    It should not be used for actual medical diagnosis without professional consultation.
    
    ---
    
    **Built with ❤️ for healthcare innovation**
    
    [GitHub Repository](https://github.com/Palak985/Medical-Disease-Prediction)
    """)

st.divider()
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>© 2026 Medical Disease Prediction System | All Rights Reserved</p>", 
           unsafe_allow_html=True)