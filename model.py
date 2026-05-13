import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

print("Loading dataset...")
df = pd.read_csv("chess_dataset_v2.csv")

X = df.drop("score", axis=1)
y = df["score"]

print("Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)



print("Creating model...")
model = MLPRegressor(
    hidden_layer_sizes=(256, 128, 64), 
    activation='relu',
    solver='adam',
    max_iter=1000,                     
    learning_rate_init=0.001,          
    early_stopping=True,               
    n_iter_no_change=10, 
    random_state=42,
    verbose=True
)

print("Training model...")
model.fit(X_train, y_train)

print("Testing model...")
predictions = model.predict(X_test)


mse = mean_squared_error(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("-" * 30)
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R-squared Score (R2): {r2:.4f} ")
print("-" * 30)

print("Saving model...")
with open("chess_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Model saved as chess_model.pkl")