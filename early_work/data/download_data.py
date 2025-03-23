import yfinance as yf

# Define the ticker symbol and date range
ticker = "TSLA"
start_date = "2023-01-01"
end_date = "2025-03-28"

# Download the stock data
tsla_data = yf.download(ticker, start=start_date, end=end_date)

# Save the data to a CSV file
tsla_data.to_csv('../data/raw/tsla_stock_data.csv')

print(f"Data for {ticker} downloaded and saved to 'data/raw/tsla_stock_data.csv'")
