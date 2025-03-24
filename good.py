import torch
import numpy as np
import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from datetime import date
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal
import sys


# Load your trained model
class GRUModel(torch.nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = torch.nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = torch.nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        out = self.linear(out[:, -1, :])
        return out

# --- Parameters ---
input_size = 5
hidden_size = 128
num_layers = 3
dropout = 0.2
sequence_length = 30  # Use last 30 days to predict next 5
forecast_steps = 5
features = ['Open', 'High', 'Low', 'Close', 'Volume']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name = 'output.pth'
#model_name = 'best.pth'
# --- Load model ---
model = GRUModel(input_size, hidden_size, num_layers, dropout).to(device)
model.load_state_dict(torch.load('./'+ model_name))  # Replace with actual saved file
model.eval()

# --- Download latest data ---
nyse = mcal.get_calendar('NYSE')
all_trading_days = nyse.valid_days(end_date='2025-03-28', start_date="2021-01-04")
end_date = all_trading_days[-6].strftime('%Y-%m-%d')  # 5 days ago = index -6 (since it's inclusive)

start_date = '2021-01-03'#end_date is is the end of the 5 days
df = yf.download('TSLA', start=start_date, end=end_date)
df = df[features]

# --- Scale using only past data ---
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)

# --- Prepare last 30 days for prediction ---
last_seq = scaled_data[-sequence_length:]  # Shape: (30, 5)
forecasted_opens = []

with torch.no_grad():
    for _ in range(forecast_steps):
        input_seq = torch.tensor(last_seq, dtype=torch.float32).unsqueeze(0).to(device)  # (1, 30, 5)
        pred = model(input_seq).cpu().numpy()[0, 0]
        forecasted_opens.append(pred)

        # Shift and update the last_seq with predicted Open and repeated last values for other features
        last_known = last_seq[-1]
        new_row = np.array([pred, *last_known[1:]])  # keep High, Low, Close, Volume the same
        last_seq = np.roll(last_seq, shift=-1, axis=0)
        last_seq[-1] = new_row

# --- Inverse transform just the Open values ---
forecast_array = np.array(forecasted_opens).reshape(-1, 1)
pad = np.zeros((forecast_array.shape[0], input_size - 1))  # fill other columns with 0s
full_forecast_scaled = np.concatenate([forecast_array, pad], axis=1)
forecast_open_prices = scaler.inverse_transform(full_forecast_scaled)[:, 0]

# --- Print results ---
# Generate next 5 NYSE trading days after the end_date
forecast_dates = nyse.valid_days(start_date=pd.to_datetime(end_date) + pd.Timedelta(days=1), end_date='2025-12-31')[:forecast_steps]

print("\n📅 Next 5 Predicted Open Prices:")
for date, price in zip(forecast_dates, forecast_open_prices):
    print(f"{date.strftime('%Y-%m-%d')}: ${price:.2f}")

# --- Plot forecast ---
plt.plot(range(1, 6), forecast_open_prices, marker='o', label='Predicted Open')
plt.title("Tesla Open Price Forecast (Next 5 Days)")
plt.xlabel("Day")
plt.ylabel("Price ($)")
plt.grid(True)
plt.legend()
plt.show()

# TODO fix logic
# --- Decision-making function ---
def trading_decision(prices):
    current_price = prices[0]
    future_avg = np.mean(prices[1:])
    print(f"\n Decision Logic:\nCurrent: ${current_price:.2f}, Future Avg: ${future_avg:.2f}")

    if future_avg > current_price * 1.02:
        return "BUY - Price is expected to rise."
    elif future_avg < current_price * 0.98:
        return "SELL - Price is expected to drop."
    else:
        return "HOLD - Price is stable."

decision = trading_decision(forecast_open_prices)
print(f"\n Recommendation: {decision}")
sys.exit()

