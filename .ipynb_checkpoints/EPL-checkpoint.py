# 1. Import libraries
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

# 2. Data Preparation
# Make copies to avoid SettingWithCopyWarning
X_train_clean = X_train.copy()
X_test_clean = X_test.copy()

# Encode categorical variables
categorical_cols = ['HomeTeam', 'AwayTeam']
le = LabelEncoder()

for col in categorical_cols:
    X_train_clean.loc[:, col] = le.fit_transform(X_train[col].astype(str))
    X_test_clean.loc[:, col] = le.transform(X_test[col].astype(str))

# Identify numeric columns for scaling
numeric_cols = X_train_clean.select_dtypes(include=np.number).columns.tolist()

# Handle missing values
imputer = SimpleImputer(strategy='mean')
X_train_clean[numeric_cols] = imputer.fit_transform(X_train_clean[numeric_cols])
X_test_clean[numeric_cols] = imputer.transform(X_test_clean[numeric_cols])

# Scale features
scaler = StandardScaler()
X_train_scaled = X_train_clean.copy()
X_test_scaled = X_test_clean.copy()

X_train_scaled[numeric_cols] = scaler.fit_transform(X_train_clean[numeric_cols])
X_test_scaled[numeric_cols] = scaler.transform(X_test_clean[numeric_cols])

# Encode target variable
y_train_encoded = y_train.map({'H': 0, 'D': 1, 'A': 2}).astype(int)
y_test_encoded = y_test.map({'H': 0, 'D': 1, 'A': 2}).astype(int)

# 3. Model Initialization
models = {
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=100, random_state=42, eval_metric='mlogloss'),
    "Logistic Regression": LogisticRegression(multi_class='multinomial', 
                                           solver='lbfgs', 
                                           max_iter=1000)
}

# 4. Model Training and Evaluation
results = {}
for name, model in models.items():
    try:
        if name == "Logistic Regression":
            model.fit(X_train_scaled[numeric_cols], y_train_encoded)
            y_pred = model.predict(X_test_scaled[numeric_cols])
        else:
            model.fit(X_train_clean[numeric_cols], y_train_encoded)
            y_pred = model.predict(X_test_clean[numeric_cols])
        
        results[name] = {
            'accuracy': accuracy_score(y_test_encoded, y_pred),
            'report': classification_report(y_test_encoded, y_pred, 
                                          target_names=['Home', 'Draw', 'Away'])
        }
    except Exception as e:
        print(f"Error with {name}: {str(e)}")
        continue

# 5. Display Results
for model_name, metrics in results.items():
    print(f"\n{model_name} Performance:")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print("Classification Report:")
    print(metrics['report'])

# 6. Feature Importance
if "Random Forest" in models:
    print("\nRandom Forest Feature Importances:")
    rf_importances = pd.DataFrame({
        'Feature': numeric_cols,  # Only show numeric features
        'Importance': models["Random Forest"].feature_importances_
    }).sort_values('Importance', ascending=False)
    print(rf_importances.head(10))