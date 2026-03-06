import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("data/weather_classification_data.csv")

# Buat encoder
le = LabelEncoder()

# Encode semua kolom teks
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# Feature dan target
X = df.drop("Weather Type", axis=1)
y = df["Weather Type"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model AI
model = RandomForestClassifier()

# Training
model.fit(X_train, y_train)

# Simpan model
joblib.dump(model, "weather_model.pkl")

print("Model berhasil dibuat!")