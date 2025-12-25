import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import joblib

# Load dataset
data = pd.read_csv(r"D:\Projects\Future-Health-Prediction\data\heart_disease.csv")

# Replace spaces in column names
data.columns = [col.strip().replace(" ", "_") for col in data.columns]

# Encode all object/string columns
for col in data.select_dtypes(include='object').columns:
    data[col] = data[col].astype('category').cat.codes

# Features and label
X = data.drop("Heart_Disease_Status", axis=1)
y = data["Heart_Disease_Status"]

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Build ANN
model = Sequential([
    Dense(16, activation='relu', input_dim=X_train.shape[1]),
    Dense(8, activation='relu'),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
model.fit(X_train, y_train, epochs=100, batch_size=10, validation_split=0.2)

# Save model and scaler
model.save("model/heart_ann.h5")
joblib.dump(scaler, "model/scaler_heart.pkl")

print("Heart model and scaler saved successfully.")


