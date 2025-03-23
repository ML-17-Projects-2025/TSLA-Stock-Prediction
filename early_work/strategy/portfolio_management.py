class PortfolioManager:
    def __init__(self, initial_cash):
        self.cash = initial_cash
        self.shares = 0

    def buy(self, price, amount):
        cost = price * amount
        if self.cash >= cost:
            self.cash -= cost
            self.shares += amount
        else:
            raise ValueError("Insufficient funds")

    def sell(self, price, amount):
        if self.shares >= amount:
            self.shares -= amount
            self.cash += price * amount
        else:
            raise ValueError("Insufficient shares")
