import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

print("Loading dataset...")

df = pd.read_csv("chess_dataset.csv")

X = df.drop("score", axis=1)
y = df["score"]


print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


print("Scaling dataset...")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


print("Creating model...")
model = MLPRegressor(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    solver='adam',
    max_iter=200,
    random_state=42,
    verbose=True
)


print("Training model...")
model.fit(X_train, y_train)


print("Testing model...")
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print("Mean Squared Error:", mse)


print("Saving model...")
with open("chess_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as chess_model.pkl")

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)
print("Scaler saved as scaler.pkl")