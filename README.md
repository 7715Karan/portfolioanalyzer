Portfolio Analyzer

A Python-based Object-Oriented Programming (OOP) tool to analyze and optimize your equity portfolio using real-time market data fetched via the Yahoo Finance API (`yfinance`). This project is structured for daily development, adding one analysis method at a time.

Features Implemented

1. Portfolio Summary
- Displays statistical summary (mean, count, std, etc.) of your portfolio DataFrame.

2. Total PnL Calculation
- Calculates and prints:
  - Total Invested Value
  - Total Current Value
  - Total Profit & Loss (PnL)

3. Fetch Historical Data
- Automatically adds `.NS` suffix for NSE stocks.
- Downloads last 6 months of daily historical data for all portfolio tickers.
- Returns closing price DataFrame.

4. Portfolio Volatility
- Computes portfolio volatility based on daily returns and weight allocation.
- Uses historical close price data.

5. Visualization:
- Pie Diagram for assest allocation
- Correlation of stock with nifty Index
- Stock Volatility Heatmap
