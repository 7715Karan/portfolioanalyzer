import requests
import pandas as pd
import dash
from dash import dcc,html,dash_table
from dash.dependencies import Input,Output
import plotly.graph_objects as go
import yfinance as yf
import numpy as np


class PortfolioAnalyser:
    def __init__(self,portfolio_df):
        self.portfolio = portfolio_df
        self.tickers = []
        

    def show_summary(self):
        print("Portfolio Summary:")
        print(self.portfolio.describe())
        self.close_price = pd.DataFrame()

    def calculate_total_PnL(self):
        total_invested = self.portfolio['Invested'].sum()
        total_current = self.portfolio['Cur. val'].sum()
        total_pnl = total_current - total_invested
        

        print(f"Total Invested Value: ₹ {total_invested}")
        print(f"Total Current Value: ₹{total_current}")
        print(f"Total PnL: ₹{total_pnl}")

    def fetch_data(self,period = "6mo",interval = '1d'):
        close_price=pd.DataFrame()
        self.tickers = self.portfolio['Instrument'].dropna().apply(lambda x: str(x).strip() + ".NS").tolist()
        data = yf.download(self.tickers,period = period,interval=interval,group_by = 'ticker',auto_adjust=True)
        for ticker in self.tickers:
            name = ticker.replace('.NS','')
            close_price[name] = data[ticker]['Close']

        return close_price
    
    def calculate_portfolio_volatility(self):
        close_price = self.fetch_data()
        if close_price.empty:
            print("No data available for the given tickers.")
            return None
        
        daily_returns = close_price.pct_change().dropna()
        total_current = self.portfolio['Cur. val'].sum()
        weights = self.portfolio.set_index('Instrument')['Cur. val'] / total_current

        weights = weights[[ col for col in close_price.columns if col in weights.index ]]
        portfolio_returns = daily_returns[weights.index].dot(weights)
        volatility = portfolio_returns.std()
        annualized_volatility = portfolio_returns.std() * (252 ** 0.5)

        print(f"daily_volatility;{volatility}")
        print(f"annualized_volatility;{annualized_volatility}")

    def calculate_diversification_score(self,period = "6mo", interval = "1d"):
        close_price = self.fetch_data(period = period,interval = interval)

        close_price = close_price.dropna(axis = 1,how = 'any')
        if close_price.shape[1] < 2:
            print('not enough data to calculate')
            return None
        daily_return = close_price.pct_change().dropna()
        corr_matrix = daily_return.corr()
        avg_corr = (corr_matrix.values[np.triu_indices_from(corr_matrix.values, k = 1)]).mean()
        diversification_score = 1 - avg_corr
        print(f"Average Asset Correlation;{avg_corr}")
        print(f"Diversification Score;{diversification_score}")

df = pd.read_csv('your_portfolio.csv')

analyzer = PortfolioAnalyser(df)
analyzer.calculate_diversification_score()



#





