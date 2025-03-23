def run_backtest(agent, data):
    initial_balance = 10000
    capital = initial_balance
    shares_owned = 0

    for index, row in data.iterrows():
        action = agent.decide(row)
        if action == "Buy":
            # Execute buy order here
            agent.buy(row['Close'], capital // row['Close'])
        elif action == "Sell":
            # Execute sell order here
            agent.sell(row['Close'], shares_owned)
        # Track portfolio value over time, etc.

    print(f"Final account balance: {capital}")
