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
    
    
    def plot_assets_allocation_and_correlation(self):
        fig,axes = plt.subplots(1,3,figsize=(14,6))
        self.portfolio.set_index('Instrument',inplace = False)['Cur. val'].plot.pie(
            ax = axes[0],
            autopct = '%1.1f%%',
            startangle = 90,
            color=plt.cm.Paired.colors
        )
        axes[0].set_title('Assets Allocation')
        axes[0].set_ylabel('')

        close_price=self.fetch_data()
        nifty_data = yf.download('^NSEI', period='6mo', interval='1d')['Close']
        nifty_returns = nifty_data.pct_change().dropna()
        returns = close_price.pct_change().dropna()
        correlation_result = {}
        for stock in returns.columns:
            aligned = pd.concat([returns[stock], nifty_returns],axis = 1).dropna()
            corelation = aligned.corr().iloc[0,1]
            correlation_result[stock] = corelation
        corelation_df = pd.DataFrame.from_dict(correlation_result,orient = 'index',columns = ['correlation'])
        sns.heatmap(corelation_df,annot=True,cmap = 'coolwarm',vmin = -1,vmax = 1,linewidth=0.5,ax=axes[1])

        axes[1].set_title(f'correlation matrix')

        rolling_volatility = returns.rolling(window=30).std()
        vol_heatmap_data = rolling_volatility.tail(1).T

        sns.heatmap(vol_heatmap_data,annot=True,cmap = "YlGnBu",linewidth=0.5,ax=axes[2])
        axes[2].set_title("30-day Rolling Volatility")
        
        plt.tight_layout()
        plt.show()
    
    
    def monte_carlo_simulation(self,num_simulation=1000,time_horizon = 252):
        close_price = self.fetch_data()
        returns = close_price.pct_change().dropna()

        available_stocks = list(returns.columns)
        portfolio_filtered = self.portfolio[self.portfolio['Instrument'].isin(available_stocks)]
        weights = portfolio_filtered.set_index('Instrument')['Cur. val']
        weights = weights / weights.sum()
        returns = returns[weights.index]
        
        portfolio_returns = returns.dot(weights)
        mean_return = portfolio_returns.mean()
        std_dev = portfolio_returns.std()

        initial_value = weights.sum()
        simulations = np.zeros((num_simulation, time_horizon))

        for i in range(num_simulation):
            daily_returns = np.random.normal(mean_return,std_dev,time_horizon)
            simulations[i] = initial_value * np.cumprod(1 + daily_returns)

        plt.figure(figsize = (12,6))
        plt.plot(simulations.T,alpha = 0.1,color = 'blue')
        plt.title(f"{num_simulation} Monte carlo simulation over {time_horizon} days")
        plt.xlabel("days")
        plt.ylabel('portfolio value')
        plt.grid(True)

        ending_values = simulations[:,-1]
        plt.axhline(y = np.percentile(ending_values,5), color = 'red',linestyle = '-',label = '5% worst case')
        plt.axhline(y = np.percentile(ending_values,95), color = 'green',linestyle = '-',label = '5% best case')
        plt.legend()
        plt.show()

        print(f"expected portfolio value after{time_horizon} days; {np.mean(ending_values)}")
        print(f"95% confidence interval: ₹{np.percentile(ending_values, 5):.2f} - ₹{np.percentile(ending_values, 95):.2f}")   
     

df = pd.read_csv('your_portfolio.csv')

analyzer = PortfolioAnalyser(df)
analyzer.calculate_diversification_score()



#





