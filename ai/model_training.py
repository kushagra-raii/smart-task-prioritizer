import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor, VotingRegressor
from sklearn.metrics import mean_absolute_error

# ----------------------------
# Generate synthetic dataset with selected features
# ----------------------------

def generate_synthetic_data(n=1000):
    np.random.seed(42)
    
    # Generate more realistic data
    data = pd.DataFrame({
        'Importance': np.random.randint(1, 6, n),  # 1-5 scale
        'Deadline_Days': np.random.randint(1, 31, n),  # 1-30 days
        'Task_Status': np.random.choice(['Not Started', 'In Progress', 'Completed'], n, p=[0.4, 0.4, 0.2]),
        'Number_of_Dependents': np.random.randint(0, 6, n)  # 0-5 dependent tasks
    })
    
    # Calculate priority based on business rules
    def calculate_priority(row):
        # Base priority from importance (1-5 scale)
        priority = row['Importance'] / 5.0
        
        # Adjust for deadline (urgency)
        if row['Deadline_Days'] <= 3:
            priority *= 1.5  # Urgent tasks get higher priority
        elif row['Deadline_Days'] <= 7:
            priority *= 1.2  # Soon due tasks get slightly higher priority
        
        # Adjust for task status
        if row['Task_Status'] == 'Not Started':
            priority *= 1.1  # Not started tasks get slightly higher priority
        elif row['Task_Status'] == 'Completed':
            priority *= 0.5  # Completed tasks get lower priority
        
        # Adjust for number of dependents
        # More dependents = higher priority
        dependent_multiplier = 1 + (row['Number_of_Dependents'] * 0.2)  # Each dependent adds 20% to priority
        priority *= dependent_multiplier
        
        # Normalize to 0-1 range
        return min(max(priority, 0.1), 1.0)
    
    # Apply priority calculation
    data['Priority'] = data.apply(calculate_priority, axis=1)
    
    return data

# Generate and display sample data
data = generate_synthetic_data()
print("### Sample Synthetic Data")
print(data.head())
print("\n### Data Statistics")
print(data.describe())

# ----------------------------
# Preprocessing
# ----------------------------

def preprocess_data(data):
    encoders = {}
    
    # Encoding categorical variables
    encoders['Task_Status'] = LabelEncoder()
    data['Task_Status'] = encoders['Task_Status'].fit_transform(data['Task_Status'])
    
    # Normalizing numerical data
    scaler = StandardScaler()
    features_to_scale = ['Importance', 'Deadline_Days', 'Number_of_Dependents']
    data[features_to_scale] = scaler.fit_transform(data[features_to_scale])
    
    return data, encoders, scaler, features_to_scale

# Preprocess the data
data, encoders, scaler, features_to_scale = preprocess_data(data)

# Prepare features and target
X = data[['Importance', 'Deadline_Days', 'Task_Status', 'Number_of_Dependents']]
y = data['Priority']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----------------------------
# Model training (using regression models)
# ----------------------------

rf_model = RandomForestRegressor(n_estimators=300, random_state=42, max_depth=10, min_samples_split=5, min_samples_leaf=3)
gb_model = GradientBoostingRegressor(n_estimators=300, random_state=42, max_depth=6, learning_rate=0.03)
et_model = ExtraTreesRegressor(n_estimators=250, random_state=42, max_depth=12)

ensemble_model = VotingRegressor(estimators=[('rf', rf_model), ('gb', gb_model), ('et', et_model)])
ensemble_model.fit(X_train, y_train)

y_pred = ensemble_model.predict(X_test)

# Since we are doing regression, let's use mean absolute error as the evaluation metric
mae = mean_absolute_error(y_test, y_pred)
print(f"Model Mean Absolute Error: {mae:.4f}")

# ----------------------------
# Save components (Model, Scaler, Encoders)
# ----------------------------

joblib.dump(ensemble_model, "task_prioritizer_model.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(encoders, "encoders.pkl")
joblib.dump(features_to_scale, "features_to_scale.pkl")

print("Model and other components have been saved!")
