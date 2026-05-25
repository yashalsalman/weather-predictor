import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

print("🚀 Starting Weather Prediction Project...")

# LOAD DATA
df = pd.read_csv("data/DailyDelhiClimateTrain.csv")

# CLEAN
df = df.dropna()
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# FEATURE ENGINEERING
df['month'] = df['date'].dt.month

# FEATURES
features = ['humidity', 'wind_speed', 'meanpressure', 'month']
target = 'meantemp'

X = df[features]
y = df[target]

# TIME SPLIT
train_size = int(len(df) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# MODEL
model = LinearRegression()
model.fit(X_train, y_train)

# PREDICT
y_pred = model.predict(X_test)

# METRICS
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"📊 RMSE: {rmse:.2f}")
print(f"📊 R2 Score: {r2:.2f}")

# SAVE PREDICTION GRAPH
plt.figure(figsize=(10,5))
plt.plot(y_test.values, label="Actual")
plt.plot(y_pred, label="Predicted")
plt.legend()
plt.title("Actual vs Predicted Temperature")
plt.savefig("outputs/prediction_plot.png")
plt.close()

# SAVE HEATMAP
plt.figure(figsize=(8,5))
sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.savefig("outputs/heatmap.png")
plt.close()

print("✅ Graphs saved in /outputs folder")