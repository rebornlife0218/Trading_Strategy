import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate


class TradingStrategy:
    def __init__(self, filepath, index_ticker, start_date, end_date, pct_of_all_stocks_to_buy=0.1):
        self.filepath = filepath
        self.index_ticker = index_ticker
        self.start_date = start_date
        self.end_date = end_date
        self.pct_of_all_stocks_to_buy = pct_of_all_stocks_to_buy
        
        self.stock_data = self.get_tickers()
        self.index = yf.download(index_ticker, start=start_date, end=end_date)['Adj Close']
        self.portfolio = self.construct_portfolio()
        self.returns = self.backtest()

    # get the n% least volatility stocks in a DataFrame over the past t months
    def get_tickers(self):
        tickers = []
        with open(self.filepath) as file:
            for line in file:
                tickers.append(line.strip())        # line.strip() 是用來去除行首和行尾的空白字符（如空格和換行符）
        data = yf.download(tickers, start=self.start_date).sort_index()['Adj Close']
        clean_data = data[data.isna().sum(axis=1) != data.shape[1]]     # 過濾掉那些股票數據都有缺失值的日期
        return clean_data

    def get_n_least_volatility(self, date, t=120, n=10):
        date = pd.to_datetime(date)
        start_date = date - pd.DateOffset(months=t)     # 計算回溯的起始日期
        data = self.stock_data.copy()[start_date:date]

        ticker_vols = {}
        for ticker in data.columns:
            # 檢查有效數據點數量
            valid_data = data[ticker].dropna()
            if len(valid_data) < 2:
                continue        # 略過樣本數量不足的股票
            return_df = np.array(valid_data.pct_change(periods=5))[5:]
            if len(return_df) < 2:      # 略過計算變異數時數據點數量不足的情況
                continue
            vol = return_df.std()
            ticker_vols[ticker] = vol       # 波動度
        
        if len(ticker_vols) == 0:       # 如果 ticker_vols 為空，返回空列表。
            return []
        sorted_tickers = sorted(ticker_vols.items(), key=lambda x: x[1])        # .items() 返回字典的鍵值對 tuple 列表，再按 tuple 的第二個元素（即 vol）從小到大排列。
        return [x[0] for x in sorted_tickers[:n]]

    def construct_portfolio(self):
        portfolio = pd.DataFrame(data=np.zeros([len(self.stock_data), len(self.stock_data.columns)]), 
                                 index=self.stock_data.index, columns=self.stock_data.columns)
        num_stocks = int(self.stock_data.shape[1] // (1 / self.pct_of_all_stocks_to_buy))       # 分子是股票數量，分母是每個股票所佔的比例。
        weight = 1 / num_stocks
        current_month = pd.to_datetime(self.start_date).month - 1

        for date in self.stock_data[self.start_date:self.end_date].index:
            if date.month != current_month:     # 查看當前日期是否進入新的月份
                stocks_in_portfolio = self.get_n_least_volatility(date=date, n=num_stocks)
                current_month = date.month
            # 遍歷選擇的股票，將其權重設定到 portfolio 中的對應日期和股票代號位置。
            for ticker in stocks_in_portfolio:
                portfolio.loc[date, ticker] = weight
        return portfolio[self.start_date:self.end_date]

    def backtest(self):
        stock_data = self.stock_data[self.start_date:self.end_date]
        portfolio = self.portfolio[self.start_date:self.end_date]
        index = self.index[self.start_date:self.end_date]

        # 計算策略每日報酬
        strategy_returns = (portfolio * stock_data.pct_change()).sum(axis=1)
        strategy_cumulative_returns = (1 + strategy_returns).cumprod() - 1

        # 計算指數每日報酬
        index_returns = index.pct_change()
        index_cumulative_returns = (1 + index_returns).cumprod() - 1

        # 計算累積百分比報酬
        strategy_cumulative_percent_returns = strategy_cumulative_returns * 100
        index_cumulative_percent_returns = index_cumulative_returns * 100

        returns = pd.DataFrame({
            'daily_returns': strategy_returns,
            'index_returns': index_returns,
            'cumulative_returns': strategy_cumulative_returns,
            'index_cumulative_returns': index_cumulative_returns,
            'cumulative_%_returns': strategy_cumulative_percent_returns,
            'index_cumulative_%_returns': index_cumulative_percent_returns
        })

        returns = returns.fillna(0)     # 填補缺失值，避免計算時出錯。
        return returns

    # 計算 Annual Return
    def calculate_annual_return(self, returns, column_name):
        num_years = len(returns) / 252
        cumulative_return = returns[column_name].iloc[-1] + 1
        annual_return = (cumulative_return ** (1 / num_years)) - 1
        return round(annual_return * 100, 2)        # 轉換為百分比並四捨五入到兩位小數
    
    # 計算 Max Drawdown
    def calculate_drawdown(self, cumulative_returns):
        cumulative_returns = cumulative_returns.copy()
        cumulative_returns += 1
        running_max = np.maximum.accumulate(cumulative_returns.dropna())
        running_max[running_max < 1] = 1        # 為了避免在計算回撤時出現錯誤，因為小於 1 的值可能是因為缺乏數據或其他原因。
        drawdown = (cumulative_returns / running_max - 1) * 100
        return drawdown.min(), drawdown

    # 計算 Win Rate
    def calculate_win_rate(self, daily_returns):
        num_positive_days = (daily_returns > 0).sum()
        win_rate = num_positive_days / len(daily_returns)
        return round(win_rate * 100, 2)     # 轉換為百分比並四捨五入到兩位小數

    def print_metrics(self):
        annual_return_1 = self.calculate_annual_return(self.returns, 'cumulative_returns')
        annual_return_2 = self.calculate_annual_return(self.returns, 'index_cumulative_returns')

        sharpe1 = round(self.returns['daily_returns'].mean()/ self.returns['daily_returns'].std() * np.sqrt(252), 2)       # 未設無風險利率，暫定為0
        sharpe2 = round(self.returns['index_returns'].mean()/ self.returns['index_returns'].std() * np.sqrt(252), 2)

        drawdown_1, _ = self.calculate_drawdown(self.returns['cumulative_returns'])
        drawdown_2, _ = self.calculate_drawdown(self.returns['index_cumulative_returns'])
        drawdown_1 = round(drawdown_1, 2)
        drawdown_2 = round(drawdown_2, 2)

        win_rate_1 = self.calculate_win_rate(self.returns['daily_returns'])
        win_rate_2 = self.calculate_win_rate(self.returns['index_returns'])

        data = {
            "": ["Least Volatility Stocks", "Index"],       # ""：空字符串鍵，對應的值是 ["Least Volatility Stocks", "Index"]，表示兩個標題行。
            "Annual Return (%)": [annual_return_1, annual_return_2],
            "Sharpe Ratio": [sharpe1, sharpe2],
            "Max Drawdown": [drawdown_1, drawdown_2],
            "Win Rate (%)": [win_rate_1, win_rate_2]
        }

        df = pd.DataFrame(data)
        table = tabulate(df, headers='keys', tablefmt='grid', showindex=False)      # 使用字典的鍵作為表格的標題
        print(table)

    def plot_returns(self):
        plt.figure(figsize=(12, 6))
        plt.plot(self.returns['cumulative_%_returns'], label='Cumulative % Returns of the Least Volatility Stocks', color='forestgreen', linewidth=2, linestyle='-')
        plt.plot(self.returns['index_cumulative_%_returns'], label='Cumulative % Returns of Index', color='firebrick', linewidth=2, linestyle='-')
        plt.title('Cumulative Strategy Returns vs Cumulative Index Returns', fontsize=17)
        plt.xlabel('Date', fontsize=15)
        plt.ylabel('Cumulative Returns (%)', fontsize=15)
        plt.legend()
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
        plt.show()

        _, drawdown_1 = self.calculate_drawdown(self.returns['cumulative_returns'])     # 返回一個完整的 drawdown 序列，另一個是計算策略的回撤的 drawdown_1
        _, drawdown_2 = self.calculate_drawdown(self.returns['index_cumulative_returns'])       # 返回一個完整的 drawdown 序列，另一個是計算指數的回撤的 drawdown_2
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.plot(drawdown_1.index, drawdown_1.values, color='forestgreen', label='Drawdown of the Least Volatility Stocks')
        ax.fill_between(drawdown_1.index, drawdown_1.values, color='forestgreen', alpha=0.4)

        ax.plot(drawdown_2.index, drawdown_2.values, color='firebrick', label='Drawdown of Index')
        ax.fill_between(drawdown_2.index, drawdown_2.values, color='firebrick', alpha=0.4)

        ax.set_title('Comparison of Drawdowns', fontsize=17)
        ax.set_xlabel('Date', fontsize=15)
        ax.set_ylabel('Drawdown(%)', fontsize=15)
        ax.tick_params(axis='x', labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend()

        plt.tight_layout()      # 自動調整圖表佈局，以避免標籤或內容被遮擋。
        plt.show()

if __name__ == '__main__':
    # 取得資料
    filepath = './上市上櫃tickers.txt'
    index_ticker = "^TWII"
    start_date = "2014-06-30"
    end_date = "2024-06-30"

    # 執行策略
    strategy = TradingStrategy(filepath, index_ticker, start_date, end_date, pct_of_all_stocks_to_buy=0.1)
    strategy.print_metrics()
    strategy.plot_returns()
