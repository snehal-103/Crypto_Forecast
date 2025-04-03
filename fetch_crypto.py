import yfinance as yf

symbols = ["BTC-USD", "ETH-USD", "DOGE-USD"]

for symbol in symbols:
    crypto = yf.Ticker(symbol)
    history = crypto.history(period="1d")

    if not history.empty:
        price = round(history["Close"].iloc[-1], 2)
    else:
        price = "N/A"

    print(f"{symbol} Latest Price: {price}")
