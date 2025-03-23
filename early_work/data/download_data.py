import yfinance as yf

# Define the ticker symbol and date range
ticker = "TSLA"
start_date = "2023-01-01"
end_date = "2025-03-28"

# Download the stock data
tsla_data = yf.download(ticker, start=start_date, end=end_date)

# Remove rows where 'ticker' or 'TSLA' appear (if there are any such rows)
tsla_data = tsla_data[~tsla_data.isin(['ticker', 'TSLA']).any(axis=2)]

# Save the data to a CSV file
tsla_data.to_csv('../data/raw/tsla_stock_data.csv')

print(f"Data for {ticker} downloaded and saved to 'data/raw/tsla_stock_data.csv'")


