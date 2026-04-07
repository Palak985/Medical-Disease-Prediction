"""
Level 3: Advanced Model Optimization
Hyperparameter Tuning, Cross-Validation, and Model Comparison
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                             roc_auc_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🚀 LEVEL 3: ADVANCED MODEL OPTIMIZATION")
print("="*70)

# ============= LOAD AND PREPARE DATA =============
print("\n📊 Loading and preparing data...")
df = pd.read_csv('dataset/Diabetes.csv')

# Handle missing values
zero_cols = ['Diastolic blood pressure', 'Triceps skin fold thickness', 
             '2-Hour serum insulin', 'Body mass index']
df[zero_cols] = df[zero_cols].replace(0, np.nan)
df[zero_cols] = df[zero_cols].fillna(df[zero_cols].median())

X = df.drop('Class variable', axis=1)
y = (df['Class variable'] == 'YES').astype(int)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test")

# ============= HYPERPARAMETER TUNING =============
print("\n" + "="*70)
print("🔧 HYPERPARAMETER TUNING RESULTS")
print("="*70)

# Dictionary to store best models
best_models = {}

# -------- 1. LOGISTIC REGRESSION --------
print("\n1️⃣ Logistic Regression Tuning...")
lr_params = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs', 'liblinear'],
    'max_iter': [1000]
}

lr_grid = GridSearchCV(LogisticRegression(random_state=42), lr_params, cv=5, scoring='f1_weighted', n_jobs=-1)
lr_grid.fit(X_train_scaled, y_train)

print(f"   Best Params: {lr_grid.best_params_}")
print(f"   Best CV Score: {lr_grid.best_score_:.4f}")

best_models['Logistic Regression'] = lr_grid.best_estimator_

# -------- 2. DECISION TREE --------
print("\n2️⃣ Decision Tree Tuning...")
dt_params = {
    'max_depth': [3, 5, 7, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'criterion': ['gini', 'entropy']
}

dt_grid = GridSearchCV(DecisionTreeClassifier(random_state=42), dt_params, cv=5, scoring='f1_weighted', n_jobs=-1)
dt_grid.fit(X_train_scaled, y_train)

print(f"   Best Params: {dt_grid.best_params_}")
print(f"   Best CV Score: {dt_grid.best_score_:.4f}")

best_models['Decision Tree'] = dt_grid.best_estimator_

# -------- 3. RANDOM FOREST --------
print("\n3️⃣ Random Forest Tuning...")
rf_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}

rf_grid = RandomizedSearchCV(RandomForestClassifier(random_state=42), rf_params, cv=5, 
                             scoring='f1_weighted', n_iter=15, n_jobs=-1, random_state=42)
rf_grid.fit(X_train_scaled, y_train)

print(f"   Best Params: {rf_grid.best_params_}")
print(f"   Best CV Score: {rf_grid.best_score_:.4f}")

best_models['Random Forest'] = rf_grid.best_estimator_

# -------- 4. GRADIENT BOOSTING --------
print("\n4️⃣ Gradient Boosting Tuning...")
gb_params = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}

gb_grid = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), gb_params, cv=5, 
                             scoring='f1_weighted', n_iter=15, n_jobs=-1, random_state=42)
gb_grid.fit(X_train_scaled, y_train)

print(f"   Best Params: {gb_grid.best_params_}")
print(f"   Best CV Score: {gb_grid.best_score_:.4f}")

best_models['Gradient Boosting'] = gb_grid.best_estimator_

# -------- 5. XGBOOST --------
print("\n5️⃣ XGBoost Tuning...")
xgb_params = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.7, 0.8, 0.9],
    'colsample_bytree': [0.7, 0.8, 0.9]
}

xgb_grid = RandomizedSearchCV(XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), 
                               xgb_params, cv=5, scoring='f1_weighted', n_iter=15, n_jobs=-1, random_state=42)
xgb_grid.fit(X_train_scaled, y_train)

print(f"   Best Params: {xgb_grid.best_params_}")
print(f"   Best CV Score: {xgb_grid.best_score_:.4f}")

best_models['XGBoost'] = xgb_grid.best_estimator_

# ============= EVALUATION ON TEST SET =============
print("\n" + "="*70)
print("📊 TEST SET PERFORMANCE COMPARISON")
print("="*70)

results = []

for model_name, model in best_models.items():
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    results.append({
        'Model': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC-AUC': auc
    })
    
    print(f"\n{model_name}:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1-Score:  {f1:.4f}")
    print(f"  ROC-AUC:   {auc:.4f}")

# Create comparison dataframe
results_df = pd.DataFrame(results).set_index('Model')
print("\n" + "="*70)
print("📋 PERFORMANCE SUMMARY")
print("="*70)
print(results_df.round(4))

# ============= CROSS-VALIDATION ANALYSIS =============
print("\n" + "="*70)
print("🔄 CROSS-VALIDATION SCORES (5-Fold)")
print("="*70)

cv_results = {}
for model_name, model in best_models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='f1_weighted')
    cv_results[model_name] = {
        'mean': cv_scores.mean(),
        'std': cv_scores.std(),
        'scores': cv_scores
    }
    
    print(f"\n{model_name}:")
    print(f"  Scores: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"  Mean:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ============= VISUALIZATIONS =============
print("\n📊 Creating visualizations...")

# 1. Model Comparison Bar Charts
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Model Performance Comparison After Tuning', fontsize=16, fontweight='bold')

metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

for idx, metric in enumerate(metrics):
    ax = axes[idx // 3, idx % 3]
    values = results_df[metric].sort_values(ascending=False)
    bars = ax.bar(range(len(values)), values, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xticks(range(len(values)))
    ax.set_xticklabels(values.index, rotation=45, ha='right')
    ax.set_ylabel(metric)
    ax.set_ylim(0, 1)
    ax.grid(axis='y', alpha=0.3)
    ax.set_title(metric)

# Remove extra subplot
axes[1, 2].remove()

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: model_comparison.png")
plt.show()

# 2. Cross-Validation Comparison
fig, ax = plt.subplots(figsize=(12, 6))

model_names = list(cv_results.keys())
means = [cv_results[m]['mean'] for m in model_names]
stds = [cv_results[m]['std'] for m in model_names]

bars = ax.bar(model_names, means, yerr=stds, capsize=5, color=colors, alpha=0.7, edgecolor='black')

# Add value labels
for bar, mean in zip(bars, means):
    ax.text(bar.get_x() + bar.get_width()/2., mean,
            f'{mean:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax.set_ylabel('F1 Score', fontsize=12)
ax.set_title('5-Fold Cross-Validation Scores (with std dev)', fontsize=14, fontweight='bold')
ax.set_ylim(0, 1)
ax.grid(axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('cv_comparison.png', dpi=300, bbox_inches='tight')
print("✓ Saved: cv_comparison.png")
plt.show()

# 3. Best Model Confusion Matrix
best_model_name = results_df['F1-Score'].idxmax()
best_model = best_models[best_model_name]
y_pred_best = best_model.predict(X_test_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
axes[0].set_title(f'Confusion Matrix - {best_model_name}')
axes[0].set_ylabel('Actual')
axes[0].set_xlabel('Predicted')

# Classification Report Heatmap
report = classification_report(y_test, y_pred_best, target_names=['No Diabetes', 'Diabetes'], output_dict=True)
report_df = pd.DataFrame(report).transpose()

sns.heatmap(report_df[['precision', 'recall', 'f1-score']].astype(float), annot=True, 
            fmt='.3f', cmap='RdYlGn', ax=axes[1], vmin=0, vmax=1)
axes[1].set_title('Classification Report Heatmap')

plt.tight_layout()
plt.savefig('best_model_metrics.png', dpi=300, bbox_inches='tight')
print("✓ Saved: best_model_metrics.png")
plt.show()

# ============= FEATURE IMPORTANCE (Best Model) =============
if hasattr(best_model, 'feature_importances_'):
    print("\n" + "="*70)
    print("🎯 FEATURE IMPORTANCE (Best Model)")
    print("="*70)
    
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance.to_string(index=False))
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors_fi = plt.cm.viridis(np.linspace(0, 1, len(feature_importance)))
    ax.barh(feature_importance['Feature'], feature_importance['Importance'], color=colors_fi)
    ax.set_xlabel('Importance')
    ax.set_title(f'Feature Importance - {best_model_name}')
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: feature_importance.png")
    plt.show()

# ============= SAVE BEST MODELS =============
print("\n" + "="*70)
print("💾 SAVING MODELS")
print("="*70)

for model_name, model in best_models.items():
    filename = f"models/{model_name.replace(' ', '_').lower()}_tuned.pkl"
    pickle.dump(model, open(filename, 'wb'))
    print(f"✓ Saved: {filename}")

# Save scaler
pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
print("✓ Saved: models/scaler.pkl")

# ============= SUMMARY REPORT =============
print("\n" + "="*70)
print("📝 OPTIMIZATION SUMMARY")
print("="*70)

print(f"\n✅ Best Model: {best_model_name}")
print(f"   - Accuracy:  {results_df.loc[best_model_name, 'Accuracy']:.4f}")
print(f"   - F1-Score:  {results_df.loc[best_model_name, 'F1-Score']:.4f}")
print(f"   - ROC-AUC:   {results_df.loc[best_model_name, 'ROC-AUC']:.4f}")

print(f"\n📊 New models are {(results_df['Accuracy'].max() - 0.77) * 100:.1f}% better than baseline!")
print("\n🎉 Level 3 Complete! Models are optimized and ready for deployment.")
print("="*70)