import torch
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Define GRU model class (adjusting the final layer to match the saved model's structure)
class GRUModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU Layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        
        # Final layer (adjusted to match saved model's 'linear' layer)
        self.linear = nn.Linear(hidden_size, 1)  # Assuming the output is a single value (price prediction)

    def forward(self, x):
        # GRU forward pass
        out, _ = self.gru(x)
        
        # Linear layer to output the final prediction
        out = self.linear(out[:, -1, :])  # Take the output from the last time step
        return out


# Function to load the trained model
def load_model(model_path, device):
    model = GRUModel(input_size=5, hidden_size=128, num_layers=3, dropout=0.2).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))  # Load the weights from 'output.pth'
    model.eval()  # Set the model to evaluation mode
    return model

# Function to predict the next stock price using the model
def predict_price(model, data, scaler, device):
    # Prepare input data, for example, the last 'n' days of stock data
    last_data = data[-1]  # Using the last available data for prediction
    # Ensure that the last_data contains all 5 features for prediction
    last_data_scaled = scaler.transform(last_data.reshape(1, -1))  # Correctly reshape to have 5 features

    # Convert to PyTorch tensor
    input_tensor = torch.tensor(last_data_scaled, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Make the prediction
    with torch.no_grad():
        predicted_price = model(input_tensor).cpu().numpy()
    
    # Return the predicted price as a scalar (flatten the array to get the scalar value)
    return predicted_price.item()  # Use .item() to get the scalar value



# Function to calculate transaction fee
def apply_transaction_fee(amount):
    return amount * (1 - 0.01)  # 1% transaction fee

# Function to give advice based on prediction
def get_advice(current_price, predicted_price, balance, shares_owned, action_threshold=0.01, sell_percentage=0.1):
    price_diff = (predicted_price - current_price) / current_price
    buy_amount = balance * 0.1  # 10% of current balance for buy decision
    
    if price_diff > action_threshold:
        # Advice to Buy (after transaction fee)
        amount_to_invest = apply_transaction_fee(buy_amount)
        num_shares_to_buy = int(amount_to_invest // current_price)
        advice = f"Buy {num_shares_to_buy} shares for ${amount_to_invest:.2f}"
    elif price_diff < -action_threshold:
        # Advice to Sell (after transaction fee)
        sell_amount = shares_owned * sell_percentage
        amount_from_sale = sell_amount * current_price
        amount_after_fee = apply_transaction_fee(amount_from_sale)
        advice = f"Sell {int(sell_amount)} shares for ${amount_after_fee:.2f} after transaction fee"
    else:
        # Advice to Hold
        advice = "Hold your position"
    
    return advice

# Main logic to load model, make predictions, and give advice
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model_path = 'output.pth'  # Path to your saved model
    data = pd.read_csv('./early_work/data/raw/tsla_stock_data.csv')  # Correct file path
    current_price = data['Close'].iloc[-1]  # Get the last closing price
    
    # Use the same MinMaxScaler that was used in training
    scaler = MinMaxScaler(feature_range=(0, 1))
    
    # Fit the scaler to the training data
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    scaler.fit(data[features].values)  # Fit the scaler to the entire dataset
    
    model = load_model(model_path, device)  # Load the trained model
    predicted_price = predict_price(model, data[features].values, scaler, device)  # Predict future price
    
    # Assume you have a balance of $10,000 and you own 100 shares of Tesla
    balance = 10000
    shares_owned = 100  # Number of shares you currently own
    advice = get_advice(current_price, predicted_price, balance, shares_owned)  # Get advice based on prediction
    
    print(f"Current Price: ${current_price}")
    print(f"Predicted Price: ${predicted_price:.2f}")  # Print the scalar predicted value with 2 decimal places
    print(f"Advice: {advice}")

if __name__ == "__main__":
    main()
