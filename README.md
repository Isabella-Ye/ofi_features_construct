In the context of cross-asset OFI, comparing AAPL’s OFI to that of SPY helps isolate individual pressure in AAPL from systematic market-wide dynamics. 
Since actual SPY order book data was unavailable, we simulate SPY’s order flow using a stochastic model that matches AAPL’s volatility and depth characteristics. This allows us to construct a meaningful cross-asset OFI feature: CrossAsset_OFI = AAPL_OFI − SPY_OFI_simulated
This differential highlights if AAPL is experiencing stronger buy/sell pressure relative to the broader market baseline.

