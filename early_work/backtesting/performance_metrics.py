def calculate_performance(initial_balance, final_balance, trades):
    returns = (final_balance - initial_balance) / initial_balance
    print(f"Total returns: {returns * 100}%")
    # You can also calculate the Sharpe ratio, max drawdown, etc.
