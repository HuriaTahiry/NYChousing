# import pandas as pd
# import matplotlib.pyplot as plt
#
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error
# import numpy as np


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from pandas.tseries.offsets import DateOffset

# Load the data
df = pd.read_csv('data.csv', sep='\t', encoding='utf-16')

# Clean column names
df.columns = df.columns.str.strip()

# Parse the month column
df['Month of Period End'] = pd.to_datetime(df['Month of Period End'], format='%B %Y', errors='coerce')
df = df[df['Month of Period End'].notna()]
df.set_index('Month of Period End', inplace=True)
df.sort_index(inplace=True)


# Custom function to convert price strings to numbers
def convert_price(price_str):
    if pd.isna(price_str):
        return np.nan
    if isinstance(price_str, (int, float)):
        return price_str

    # Remove $ and commas
    price_str = str(price_str).replace('$', '').replace(',', '').strip()

    # Handle K suffix (thousands)
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
df.ffill(inplace=True)

# Features and target
features = ['Inventory', 'New Listings', 'Homes Sold', 'Days on Market', 'Average Sale To List']
target = 'Median Sale Price'

# Drop rows where target is still missing
df = df[df[target].notna()]

# Check if we have data
if len(df) == 0:
    print("ERROR: No valid data remaining after cleaning")
else:
    # Prepare data
    X = df[features]
    y = df[target]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=False, test_size=0.2)

    # Train model
    # Train model
    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    print(f"RMSE: ${rmse:,.2f}")

    # Accuracy estimate
    mean_price = y_test.mean()
    accuracy = (1 - rmse / mean_price) * 100
    print(f"Estimated Model Accuracy: {accuracy:.2f}%")

    # R² Score
    from sklearn.metrics import r2_score

    r2 = r2_score(y_test, preds)
    print(f"R² Score: {r2:.4f}")

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(y_test.index, y_test, label='Actual')
    plt.plot(y_test.index, preds, label='Predicted', linestyle='--')
    plt.title("Actual vs Predicted Housing Prices")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True)
    plt.show()
# ahhhhhhhhh


# 1. Prepare time-based features
df['time_index'] = (df.index - df.index.min()).days  # Days since first record
df['month'] = df.index.month
df['year'] = df.index.year

# 2. Define features and target
features = ['Inventory', 'New Listings', 'Homes Sold',
           'Days on Market', 'Average Sale To List',
           'time_index', 'month', 'year']
target = 'Median Sale Price'

# 3. Train on ALL historical data (no test split)
X = df[features]
y = df[target]
model = RandomForestRegressor(n_estimators=200)
model.fit(X, y)

# 4. Create future dataframe
last_date = df.index.max()
future_dates = [last_date + DateOffset(months=x) for x in range(1, 61)]  # 5 years
future_df = pd.DataFrame(index=future_dates)
future_df['time_index'] = (future_df.index - df.index.min()).days
future_df['month'] = future_df.index.month
future_df['year'] = future_df.index.year

# 5. Make predictions (you'll need to provide future feature values)
# Option A: Use average values for other features
for feature in ['Inventory', 'New Listings', 'Homes Sold', 'Days on Market', 'Average Sale To List']:
    future_df[feature] = df[feature].mean()

# Option B: Use the last known values (better for short-term predictions)
# for feature in ['Inventory', 'New Listings', 'Homes Sold', 'Days on Market', 'Average Sale To List']:
#     future_df[feature] = df[feature].iloc[-1]

future_predictions = model.predict(future_df[features])

# 6. Combine historical and future data
combined = pd.concat([
    df[['Median Sale Price']],
    pd.DataFrame({'Predicted Price': future_predictions}, index=future_df.index)
])

# 7. Visualize
plt.figure(figsize=(14, 7))
plt.plot(combined.index, combined['Median Sale Price'], label='Historical', color='blue')
plt.plot(combined.index, combined['Predicted Price'], label='Predicted', color='red', linestyle='--')
plt.axvline(x=last_date, color='gray', linestyle=':', label='Prediction Start')
plt.title('5-Year Housing Price Forecast')
plt.xlabel('Year')
plt.ylabel('Median Price ($)')
plt.legend()
plt.grid(True)
plt.show()

#
# # Correct read with tab separator and proper encoding
# df = pd.read_csv('data.csv', encoding='utf-16', sep='\t')
#
# # Convert 'Median Sale Price' and other key columns to numeric
# df['Median Sale Price'] = df['Median Sale Price'].replace('[\$,]', '', regex=True).astype(float)
#
# cols_to_clean = [
#     'Median Sale Price', 'Median Sale Price MoM', 'Median Sale Price YoY',
#     'Homes Sold', 'Homes Sold MoM', 'Homes Sold YoY',
#     'New Listings', 'New Listings MoM', 'New Listings YoY',
#     'Inventory', 'Inventory MoM', 'Inventory YoY',
#     'Days on Market', 'Days on Market MoM', 'Days on Market YoY',
#     'Average Sale To List', 'Average Sale To List MoM', 'Average Sale To List YoY'
# ]
#
# for col in cols_to_clean:
#     df[col] = df[col].replace('[\$,%,]', '', regex=True)
#     df[col] = pd.to_numeric(df[col], errors='coerce')
#
#
# # Clean column names
# df.columns = df.columns.str.strip()
#
# # Print to verify all columns are now separate
# print(df.columns.tolist())
#
# # Convert date column
# df['Month of Period End'] = pd.to_datetime(df['Month of Period End'])
#
# # Print a few rows to confirm it worked
# print(df.head())



#
# # Filter for a specific region
# df = df[df['Region'] == 'New York, NY']
#
# # Set date as index
# df.set_index('Month of Period End', inplace=True)
#
# # Ensure numeric types (remove % and convert)
# for col in df.columns:
#     if df[col].dtype == 'object' and df[col].str.contains('%').any():
#         df[col] = df[col].str.replace('%', '').astype(float) / 100
#
# # Drop or fill missing values
# df.fillna(method='ffill', inplace=True)  # forward fill is often good for time series
#
#
# df['Median Sale Price'].plot(figsize=(12, 6), title='Median Sale Price Over Time')
# plt.xlabel('Date')
# plt.ylabel('Price ($)')
# plt.show()
#
# df = pd.read_csv('data.csv')
# df['date'] = pd.to_datetime(df['date'])
# df.set_index('date', inplace=True)
# df = df.sort_index()
#
# print(df.columns.tolist())
