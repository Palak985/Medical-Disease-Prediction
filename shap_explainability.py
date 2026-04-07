"""
SHAP Explainability Module for Model Interpretation
Interactive explainability dashboard using SHAP values
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Model Explainability",
    page_icon="🔍",
    layout="wide"
)

st.markdown("""
    <style>
    .main-header {
        color: #1f77b4;
        font-size: 2.5em;
        font-weight: bold;
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-header">🔍 Model Explainability & Interpretation</div>', unsafe_allow_html=True)
st.markdown("**Understand why the model makes predictions using SHAP (SHapley Additive exPlanations)**")
st.divider()

# Load and prepare data
@st.cache_resource
def load_and_prepare_data():
    df = pd.read_csv('dataset/Diabetes.csv')
    
    # Handle missing values
    zero_cols = ['Diastolic blood pressure', 'Triceps skin fold thickness', 
                 '2-Hour serum insulin', 'Body mass index']
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    df[zero_cols] = df[zero_cols].fillna(df[zero_cols].median())
    
    X = df.drop('Class variable', axis=1)
    y = (df['Class variable'] == 'YES').astype(int)
    
    # Split and scale
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Random Forest model (best performance)
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train_scaled, y_train)
    
    return {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'X_train_scaled': X_train_scaled,
        'X_test_scaled': X_test_scaled,
        'y_train': y_train,
        'y_test': y_test,
        'scaler': scaler,
        'feature_names': X.columns.tolist()
    }

@st.cache_resource
def compute_shap_values(data):
    """Compute SHAP values for the model"""
    explainer = shap.TreeExplainer(data['model'])
    shap_values = explainer.shap_values(data['X_test_scaled'])
    return explainer, shap_values

# Load data
data = load_and_prepare_data()
model = data['model']
feature_names = data['feature_names']

# Compute SHAP values
explainer, shap_values = compute_shap_values(data)

# Sidebar navigation
st.sidebar.title("📚 Navigation")
page = st.sidebar.radio(
    "Select Analysis Type:",
    ["🎯 Feature Importance", "📊 Summary Plots", "👤 Individual Prediction", "💡 Feature Impact"]
)

# ============= PAGE 1: FEATURE IMPORTANCE =============
if page == "🎯 Feature Importance":
    st.header("Global Feature Importance (SHAP)")
    
    st.markdown("""
    **Feature Importance** shows which features have the most impact on predictions across all patients.
    
    - **Red shifts** = Feature pushes prediction toward diabetes
    - **Blue shifts** = Feature pushes prediction away from diabetes
    - **Larger dots** = Stronger impact
    """)
    
    st.divider()
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Mean |SHAP| Values")
        st.write("Average absolute impact of each feature")
        
        # Bar plot of mean absolute SHAP values
        mean_shap = np.abs(shap_values[1]).mean(axis=0)
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': mean_shap
        }).sort_values('Importance', ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = plt.cm.viridis(np.linspace(0, 1, len(feature_importance_df)))
        ax.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color=colors)
        ax.set_xlabel('Mean |SHAP| Value', fontsize=12)
        ax.set_title('Feature Importance for Diabetes Prediction', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        st.pyplot(fig)
    
    with col2:
        st.subheader("Feature Importance Table")
        st.dataframe(feature_importance_df.reset_index(drop=True), use_container_width=True, hide_index=True)
    
    st.divider()
    
    # SHAP summary plot (dot plot)
    st.subheader("SHAP Summary Plot (Dot Plot)")
    st.write("Shows how each feature value impacts the model output")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    try:
        shap.summary_plot(shap_values[1], data['X_test_scaled'], 
                         feature_names=feature_names, plot_type="dot", ax=ax, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not generate plot: {e}")

# ============= PAGE 2: SUMMARY PLOTS =============
elif page == "📊 Summary Plots":
    st.header("SHAP Summary Visualizations")
    
    st.markdown("""
    Multiple ways to visualize SHAP values and understand model behavior:
    
    1. **Bar Plot** - Importance of features
    2. **Violin Plot** - Distribution of SHAP values per feature
    3. **Dependence Plot** - Feature value vs SHAP value relationship
    """)
    
    st.divider()
    
    plot_type = st.radio("Select Plot Type:", ["Bar Plot", "Violin Plot", "Dependence Plot"])
    
    if plot_type == "Bar Plot":
        st.subheader("Feature Importance (Bar Plot)")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            shap.summary_plot(shap_values[1], data['X_test_scaled'],
                            feature_names=feature_names, plot_type="bar", ax=ax, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")
    
    elif plot_type == "Violin Plot":
        st.subheader("SHAP Values Distribution (Violin Plot)")
        st.write("Shows the distribution of SHAP values for each feature")
        
        fig = plt.figure(figsize=(12, 8))
        try:
            shap.summary_plot(shap_values[1], data['X_test_scaled'],
                            feature_names=feature_names, plot_type="violin", show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")
    
    else:  # Dependence Plot
        st.subheader("Dependence Plot")
        st.write("Shows how a feature's value affects its SHAP value (impact on prediction)")
        
        feature_select = st.selectbox("Select Feature:", feature_names)
        feature_idx = feature_names.index(feature_select)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        try:
            shap.dependence_plot(feature_idx, shap_values[1], data['X_test_scaled'],
                               feature_names=feature_names, ax=ax, show=False)
            st.pyplot(fig)
        except Exception as e:
            st.error(f"Error: {e}")

# ============= PAGE 3: INDIVIDUAL PREDICTION EXPLANATION =============
elif page == "👤 Individual Prediction Explanation":
    st.header("Explain Individual Predictions")
    
    st.markdown("""
    Select a patient from the test set to see:
    - Their prediction
    - Which features contributed most
    - How each feature pushed the prediction up or down
    """)
    
    st.divider()
    
    # Select patient
    patient_idx = st.slider("Select Patient Index:", 0, len(data['X_test']) - 1, 0)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("Patient Data")
        patient_features = data['X_test'].iloc[patient_idx]
        patient_df = pd.DataFrame({
            'Feature': feature_names,
            'Value': patient_features.values
        })
        st.dataframe(patient_df, use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("Actual Outcome")
        actual = data['y_test'].iloc[patient_idx]
        prediction = model.predict(data['X_test_scaled'])[patient_idx]
        prediction_proba = model.predict_proba(data['X_test_scaled'])[patient_idx]
        
        st.metric("Actual Label", "Diabetes" if actual == 1 else "No Diabetes")
        st.metric("Predicted Label", "Diabetes" if prediction == 1 else "No Diabetes")
        st.metric("Diabetes Probability", f"{prediction_proba[1]:.1%}")
        
        if actual == prediction:
            st.success("✅ Prediction Correct")
        else:
            st.error("❌ Prediction Incorrect")
    
    st.divider()
    
    # Force plot
    st.subheader("SHAP Force Plot")
    st.write("Shows how each feature contributes to pushing the prediction toward or away from diabetes")
    
    try:
        fig = plt.figure(figsize=(14, 4))
        shap.force_plot(explainer.expected_value[1], shap_values[1][patient_idx], 
                       data['X_test_scaled'][patient_idx], feature_names=feature_names,
                       matplotlib=True, show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not generate force plot: {e}")
    
    st.divider()
    
    # Waterfall plot
    st.subheader("SHAP Waterfall Plot")
    st.write("Cumulative contribution of each feature to the prediction")
    
    try:
        fig = plt.figure(figsize=(10, 8))
        explainer_instance = shap.Explainer(model, data['X_train_scaled'])
        shap_value = explainer_instance(data['X_test_scaled'][patient_idx:patient_idx+1])
        shap.plots.waterfall(shap_value[0], show=False)
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Could not generate waterfall plot: {e}")
    
    st.divider()
    
    # Feature contribution table
    st.subheader("Feature Contributions Breakdown")
    
    contributions = pd.DataFrame({
        'Feature': feature_names,
        'Value': data['X_test_scaled'][patient_idx],
        'SHAP Value': shap_values[1][patient_idx],
        'Impact Direction': ['↑ Increases Risk' if x > 0 else '↓ Decreases Risk' for x in shap_values[1][patient_idx]]
    }).sort_values('SHAP Value', key=abs, ascending=False)
    
    st.dataframe(contributions, use_container_width=True, hide_index=True)

# ============= PAGE 4: FEATURE IMPACT ANALYSIS =============
elif page == "💡 Feature Impact":
    st.header("Feature Impact Analysis")
    
    st.markdown("""
    Understand how changes in a specific feature impact predictions.
    
    **Scatter Plot:** Shows the relationship between a feature's value and its SHAP value
    (impact on prediction).
    """)
    
    st.divider()
    
    # Select feature
    selected_feature = st.selectbox("Select Feature to Analyze:", feature_names)
    feature_idx = feature_names.index(selected_feature)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader(f"{selected_feature} Value Distribution")
        
        fig, ax = plt.subplots(figsize=(8, 6))
        diabetic_idx = data['y_test'] == 1
        healthy_idx = data['y_test'] == 0
        
        ax.hist(data['X_test'].iloc[healthy_idx][selected_feature], alpha=0.6, 
               label='No Diabetes', bins=20, color='steelblue')
        ax.hist(data['X_test'].iloc[diabetic_idx][selected_feature], alpha=0.6, 
               label='Diabetes', bins=20, color='salmon')
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{selected_feature} Distribution')
        ax.legend()
        ax.grid(alpha=0.3)
        st.pyplot(fig)
    
    with col2:
        st.subheader(f"{selected_feature} SHAP Impact")
        
        # Feature statistics
        feature_data = data['X_test'][selected_feature]
        st.metric("Mean Value", f"{feature_data.mean():.2f}")
        st.metric("Std Dev", f"{feature_data.std():.2f}")
        st.metric("Min Value", f"{feature_data.min():.2f}")
        st.metric("Max Value", f"{feature_data.max():.2f}")
    
    st.divider()
    
    # SHAP vs Feature Value scatter
    st.subheader("SHAP Value vs Feature Value")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    scatter = ax.scatter(data['X_test_scaled'][:, feature_idx], shap_values[1][:, feature_idx],
                        c=data['y_test'], cmap='coolwarm', alpha=0.6, s=50)
    
    ax.set_xlabel(f"{selected_feature} (Scaled)", fontsize=12)
    ax.set_ylabel("SHAP Value", fontsize=12)
    ax.set_title(f"{selected_feature}: Impact on Prediction", fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Outcome (0=No Diabetes, 1=Diabetes)')
    
    st.pyplot(fig)
    
    # Interpretation
    st.info(f"""
    **Interpretation:**
    - **Positive SHAP values:** {selected_feature} increases diabetes risk
    - **Negative SHAP values:** {selected_feature} decreases diabetes risk
    - **Red dots:** Diabetic patients
    - **Blue dots:** Healthy patients
    """)

st.divider()
st.markdown("---")
st.markdown("""
### 📚 About SHAP

SHAP (SHapley Additive exPlanations) is a game-theoretic approach to explain predictions:

- **Global Explanations:** Which features matter most overall
- **Local Explanations:** Why a specific prediction was made
- **Force Plots:** Show contribution of each feature
- **Dependence Plots:** Feature value vs impact relationship

[Learn more about SHAP](https://shap.readthedocs.io/)
""")