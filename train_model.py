import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE

# Load dataset
file_path = "Depression.csv"
df = pd.read_csv(file_path)

# Define features and target
features = ["Academic Pressure", "Work Pressure", "Study Satisfaction", "Sleep Duration", "Financial Stress"]
target = "Depression"

# Encode categorical feature
label_encoders = {}
df_encoded = df.copy()
le = LabelEncoder()
df_encoded["Sleep Duration"] = le.fit_transform(df_encoded["Sleep Duration"])
label_encoders["Sleep Duration"] = le

# Drop missing values
df_encoded = df_encoded.dropna()

# Split dataset
X = df_encoded[features]
y = df_encoded[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handle imbalanced data using SMOTE
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

# Train Random Forest with hyperparameter tuning
param_grid = {"n_estimators": [50, 100, 150], "max_depth": [5, 10, 15]}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_
accuracy = best_model.score(X_test, y_test)
print(f"✅ Model Trained with Accuracy: {accuracy * 100:.2f}%")

# Save model, scaler, and encoders
joblib.dump(best_model, "model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(label_encoders, "label_enc.pkl")

print("✅ Model and encoders saved successfully!")
