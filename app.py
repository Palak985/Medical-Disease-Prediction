import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv'
columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
           'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']

df = pd.read_csv(url, names=columns)

print('Shape:', df.shape)
print('\nFirst 5 rows:')
print(df.head())
print('\nClass distribution:')
print(df['Outcome'].value_counts())

print(df.describe().round(2))

print('\nZero counts (potential missing values):')
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
print(df[zero_cols].eq(0).sum())

# Replace biologically impossible zeros with median
df[zero_cols] = df[zero_cols].replace(0, np.nan)
df[zero_cols] = df[zero_cols].fillna(df[zero_cols].median())

fig, axes = plt.subplots(2, 4, figsize=(16, 8))
features = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
            'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

for ax, feat in zip(axes.flatten(), features):
    df[df['Outcome']==0][feat].hist(ax=ax, alpha=0.6, label='No diabetes', bins=20, color='steelblue')
    df[df['Outcome']==1][feat].hist(ax=ax, alpha=0.6, label='Diabetes', bins=20, color='salmon')
    ax.set_title(feat)
    ax.legend(fontsize=8)

plt.tight_layout()
plt.show()

X = df.drop('Outcome', axis=1)
y = df['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
    'Random Forest':       RandomForestClassifier(n_estimators=100, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc
    print(f'{name}: {acc:.3f}')

    best_model = models['Random Forest']
preds = best_model.predict(X_test_scaled)

print('Classification Report (Random Forest):')
print(classification_report(y_test, preds,
      target_names=['No Diabetes', 'Diabetes']))

importances = pd.Series(
    best_model.feature_importances_,
    index=X.columns
).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
importances.plot(kind='barh', color='steelblue')
plt.title('Feature importance — Random Forest')
plt.show()

def predict_diabetes(pregnancies, glucose, blood_pressure,
                     skin_thickness, insulin, bmi, dpf, age):

    patient = pd.DataFrame([{
        'Pregnancies': pregnancies, 'Glucose': glucose,
        'BloodPressure': blood_pressure, 'SkinThickness': skin_thickness,
        'Insulin': insulin, 'BMI': bmi,
        'DiabetesPedigreeFunction': dpf, 'Age': age
    }])

    patient_scaled = scaler.transform(patient)
    prob = best_model.predict_proba(patient_scaled)[0][1]
    label = 'HIGH RISK' if prob > 0.5 else 'Low risk'
    print(f'Diabetes probability: {prob:.1%}')
    print(f'Risk level: {label}')
    return prob

# Test with first row of the dataset (known diabetic)
predict_diabetes(
    pregnancies=6, glucose=148, blood_pressure=72,
    skin_thickness=35, insulin=0, bmi=33.6,
    dpf=0.627, age=50
)