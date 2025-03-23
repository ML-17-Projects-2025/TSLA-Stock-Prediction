import pandas as pd

def add_technical_indicators(data):
    # Simple Moving Averages
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['SMA_200'] = data['Close'].rolling(window=200).mean()

    # RSI - Relative Strength Index
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    data['RSI'] = 100 - (100 / (1 + rs))

    # Label: 1 if price will go up, 0 if down
    data['Label'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    return data
