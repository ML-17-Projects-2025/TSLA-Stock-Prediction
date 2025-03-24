import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas_market_calendars as mcal
import yfinance as yf

# Define the GRU model
class GRUModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=128, num_layers=3, dropout=0.2):
        super(GRUModel, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.gru(x)
        return self.linear(out[:, -1, :])

# Load the trained model
def load_model(model_path, device):
    model = GRUModel().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Predict the next stock price
def predict_price(model, data, scaler, device):
    last_sequence_scaled = scaler.transform(data[-30:])
    input_tensor = torch.tensor(last_sequence_scaled, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        scaled_prediction = model(input_tensor).item()
    
    # Pad the scaled prediction with zeros to match the scaler's expected input shape
    padded_scaled_prediction = np.zeros((1, scaler.n_features_in_))
    padded_scaled_prediction[0, 0] = scaled_prediction  # Place the prediction in the 'Open' column

    # Inverse transform
    return scaler.inverse_transform(padded_scaled_prediction)[0, 0]

# Apply transaction fee
def apply_transaction_fee(amount, fee_rate=0.01):
    return amount * (1 - fee_rate)

# Provide trading advice
def get_advice(current_price, predicted_price, balance, shares_owned, action_threshold=0.01, sell_percentage=0.1):
    current_price = current_price.iloc[0] if isinstance(current_price, pd.Series) else current_price
    predicted_price = predicted_price.iloc[0] if isinstance(predicted_price, pd.Series) else predicted_price

    price_diff = (predicted_price - current_price) / current_price
    if price_diff > action_threshold:
        amount_to_invest = apply_transaction_fee(balance * 0.1)
        num_shares_to_buy = int(amount_to_invest // current_price)
        return f"Buy {num_shares_to_buy} shares for ${amount_to_invest:.2f}"
    
    elif price_diff < -action_threshold:
        shares_to_sell = int(shares_owned * sell_percentage)
        sale_value = apply_transaction_fee(shares_to_sell * current_price)
        return f"Sell {shares_to_sell} shares for ${sale_value:.2f} after fees"
    
    return "Hold your position"

# Main function
def main():
    model_path = 'output.pth'
    ticker = 'TSLA'
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get NYSE trading days
    nyse = mcal.get_calendar('NYSE')
    all_trading_days = nyse.valid_days(start_date="2021-01-04", end_date="2025-03-28")
    end_date = all_trading_days[-6].strftime('%Y-%m-%d')
    
    # Load stock data
    dataset = yf.download(ticker, start="2021-01-03", end=end_date)[features]
    
    # Scale data
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(dataset)
    
    # Load model and predict
    model = load_model(model_path, device)
    predicted_price = predict_price(model, scaled_data, scaler, device)
    
    # Assume balance and shares
    balance = 10_000
    shares_owned = 100
    current_price = dataset.iloc[-1]['Close']
    
    # Get trading advice
    advice = get_advice(current_price, predicted_price, balance, shares_owned)
    
    # Display results
    current_price = current_price.iloc[0] if isinstance(current_price, pd.Series) else current_price
    predicted_price = predicted_price.iloc[0] if isinstance(predicted_price, pd.Series) else predicted_price
    print(f"Current Price: ${current_price:.2f}")
    print(f"Predicted Price: ${predicted_price:.2f}")
    print(f"Advice: {advice}")

if __name__ == "__main__":
    main()
