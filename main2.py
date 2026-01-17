import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Load and clean data (same as before)
# here im loading a dataset from a CSV file. it uses tab sep and utf16 encoding,
# which is kinda weird but thats how the file was saved. then cleaning column names.
df = pd.read_csv('data.csv', sep='\t', encoding='utf-16')
df.columns = df.columns.str.strip()

# convert the "Month of Period End" into actual datetime so we can sort and use it as index.
df['Month of Period End'] = pd.to_datetime(df['Month of Period End'], format='%B %Y', errors='coerce')
df = df[df['Month of Period End'].notna()]   # drop bad date rows
df.set_index('Month of Period End', inplace=True)
df.sort_index(inplace=True)

# Custom function to convert price strings to numbers
# this fuction takes price strings like "$350K" and turn it into number like 350000.
# It also try to remove commas or other stuff the dataset might have.
def convert_price(price_str):
    if pd.isna(price_str):
        return np.nan
    if isinstance(price_str, (int, float)):
        return price_str

    price_str = str(price_str).replace('$', '').replace(',', '').strip()
    if 'K' in price_str:
        return float(price_str.replace('K', '')) * 1000
    try:
        return float(price_str)
    except:
        return np.nan

# Apply to all price columns
price_cols = ['Median Sale Price']
for col in price_cols:
    df[col] = df[col].apply(convert_price)

# Clean other numeric columns
# These columns sometimes have percents or $ signs so we remove them and convert to numbers
numeric_cols = [
    'Median Sale Price MoM', 'Median Sale Price YoY',
    'Homes Sold', 'Homes Sold MoM', 'Homes Sold YoY',
    'New Listings', 'New Listings MoM', 'New Listings YoY',
    'Inventory', 'Inventory MoM', 'Inventory YoY',
    'Days on Market', 'Days on Market MoM', 'Days on Market YoY',
    'Average Sale To List', 'Average Sale To List MoM', 'Average Sale To List YoY'
]

for col in numeric_cols:
    if df[col].dtype == 'object':
        df[col] = df[col].replace('[\$,%,]', '', regex=True)
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Forward fill missing values
# Ffill is used so missing vals get replaced with last known value
df.ffill(inplace=True)

# Select features for LSTM
# These are the features I picked becuz they impact the housing price maybe
features = ['Inventory', 'New Listings', 'Homes Sold', 'Days on Market', 'Average Sale To List']
target = 'Median Sale Price'

# Prepare data
df_model = df[features + [target]].dropna()

# Normalize the data
# scaling because LSTM models work better when everything is between 0-1 range.
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df_model[features])
y_scaled = scaler_y.fit_transform(df_model[[target]])

# Create sequences for LSTM
# this function makes the sliding windows. basically each X_seq contains 12 months of data
# to predict next month. LSTM needs sequences not single rows.
def create_sequences(X, y, time_steps=12):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:(i + time_steps)])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)

TIME_STEPS = 12  # Using 12 months as sequence length
X_seq, y_seq = create_sequences(X_scaled, y_scaled, TIME_STEPS)

# Train-test split (chronological order - no shuffling)
# Important: since it's time series we cannot shuffle the data randomly.
split_idx = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split_idx], X_seq[split_idx:]
y_train, y_test = y_seq[:split_idx], y_seq[split_idx:]

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

# Build LSTM model
# Here is the actual neural network. First LSTM returns sequences so next LSTM can read them.
# Dropout stops overfit. Dense at the end outputs one price prediction.
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(TIME_STEPS, len(features))),
    Dropout(0.2),
    LSTM(50, return_sequences=False),
    Dropout(0.2),
    Dense(25, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Early stopping callback
# this stops training when validation loss doesnt get better for many epochs
early_stop = EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)

# Train the model
# validation_split splits 20% from train data for validating
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=16,
    callbacks=[early_stop],
    verbose=1
)

# Make predictions
# predict on test data and then inverse scale so we see real prices (not normalized ones).
y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Calculate metrics
# RMSE, MAE, R2, MAPE to tell how good or bad the model is.
rmse = np.sqrt(mean_squared_error(y_test_actual, y_pred))
mae = mean_absolute_error(y_test_actual, y_pred)
r2 = r2_score(y_test_actual, y_pred)
mape = np.mean(np.abs((y_test_actual - y_pred) / y_test_actual)) * 100

print("\n" + "=" * 50)
print("LSTM MODEL PERFORMANCE")
print("=" * 50)
print(f"RMSE: ${rmse:,.2f}")
print(f"MAE: ${mae:,.2f}")
print(f"R² Score: {r2:.4f}")
print(f"MAPE: {mape:.2f}%")

# Plot training history
# showing how model loss and mae changes across epochs
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE')
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.legend()
plt.tight_layout()
plt.show()

# Plot predictions vs actual
# this graph shows the real prices vs what the model predicted
test_dates = df_model.index[TIME_STEPS + split_idx:]

plt.figure(figsize=(14, 6))
plt.plot(test_dates, y_test_actual, label='Actual Prices', linewidth=2, alpha=0.8)
plt.plot(test_dates, y_pred, label='LSTM Predictions', linewidth=2, linestyle='--', alpha=0.9)
plt.title('LSTM: Actual vs Predicted Housing Prices', fontsize=14)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Median Price ($)', fontsize=12)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Feature importance using SHAP (optional but insightful)
# I tried to use shap to see which features matter more. This part might error if shap isnt installed.
try:
    import shap

    # Create a simpler model for SHAP becuz shap sometimes struggles with big lstm models.
    simple_model = Sequential([
        LSTM(50, input_shape=(TIME_STEPS, len(features))),
        Dense(1)
    ])
    simple_model.compile(optimizer='adam', loss='mse')
    simple_model.fit(X_train, y_train, epochs=10, verbose=0)

    # Use DeepExplainer for LSTM
    explainer = shap.DeepExplainer(simple_model, X_train[:100])
    shap_values = explainer.shap_values(X_test[:100])

    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values[0].mean(axis=1), features=features, feature_names=features)
    plt.title('LSTM Feature Importance')
    plt.tight_layout()
    plt.show()
except:
    print("SHAP not installed or error occurred. Install with: pip install shap")
