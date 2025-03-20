class TradingAgent:
    def __init__(self, model):
        self.model = model
        self.cash = 10000
        self.shares = 0

    def decide(self, current_data):
        features = current_data[['SMA_50', 'SMA_200', 'RSI']].values.reshape(1, -1)
        prediction = self.model.predict(features)

        if prediction == 1 and self.cash > current_data['Close']:  # Buy
            return "Buy"
        elif prediction == 0 and self.shares > 0:  # Sell
            return "Sell"
        else:
            return "Hold"
