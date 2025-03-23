import yfinance as yf
import pandas as pd
from models.model_training import train_model
from strategy.trading_agent import TradingAgent
from backtesting.backtest import run_backtest

def main():
    # Load data (you can choose to load from processed or raw)
    tsla_data = pd.read_csv('data/processed/tsla_processed_data.csv')
    
    # Train the model
    model = train_model(tsla_data)
    
    # Initialize the trading agent with the trained model
    agent = TradingAgent(model=model)
    
    # Simulate the trading strategy (March 24-28, 2025)
    run_backtest(agent, tsla_data)

if __name__ == '__main__':
    main()
