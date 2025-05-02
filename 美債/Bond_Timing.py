import pandas as pd
import numpy as np
import time

# 讀取資料
file_path = 'TLT_23years.feather'
data = pd.read_feather(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
df = pd.DataFrame(data)
df['Return'] = df['Close'].pct_change()

dates_Raise = [
    '2004-06-30', '2004-08-10', '2004-09-21', '2004-11-10', '2004-12-14', '2005-02-02', '2005-03-22',
    '2005-05-03', '2005-06-30', '2005-08-09', '2005-09-20', '2005-11-01', '2005-12-13', '2006-01-31',
    '2006-03-28', '2006-05-10', '2006-06-29', '2015-12-17', '2016-12-14', '2017-03-15', '2017-06-14',
    '2017-12-13', '2018-03-21', '2018-06-13', '2018-09-27', '2018-12-20', '2022-03-17', '2022-05-04',
    '2022-06-16', '2022-07-28', '2022-09-22', '2022-11-03', '2022-12-15', '2023-02-02', '2023-03-23',
    '2023-05-04', '2023-07-27'
]

dates_Cut = [
    '2001-01-03', '2001-01-31', '2001-03-20', '2002-11-06', '2001-04-18', '2001-05-15', '2001-06-27', 
    '2001-08-21', '2001-09-17','2001-10-02', '2001-11-06', '2001-12-11', '2002-11-06', '2003-06-25', 
    '2007-09-18', '2007-10-31', '2007-12-11', '2008-01-22', '2008-01-30', '2008-03-18', '2008-04-30', 
    '2008-10-08', '2008-10-29', '2019-08-01', '2019-09-19', '2019-10-31', '2020-03-03', '2020-03-15', 
    '2024-09-19', '2024-11-08', '2024-12-19'
]

df['Raise'] = 0
df['Cut'] = 0
df.loc[df['Date'].isin(pd.to_datetime(dates_Raise)), 'Raise'] = 1
df.loc[df['Date'].isin(pd.to_datetime(dates_Cut)), 'Cut'] = 1

# 初始化兩個結果儲存結構
results_raise = {
    'Date': [],
    'Return': [],
    'Sharpe': [],
    'Win_Rate': [],
    'MaxDD': []
}

results_cut = {
    'Date': [],
    'Return': [],
    'Sharpe': [],
    'Win_Rate': [],
    'MaxDD': []
}

def calc_metrics_for_signal(df, signal, direction, buy_days_after, sell_days_after):
    results = {'Date': [], 'Return': []}
    trade_returns = []
    
    for idx, row in df[df[signal] == 1].iterrows():
        entry_date = row['Date']
        
        # 確保有足夠的天數進行買入和賣出
        if idx + buy_days_after < len(df) and idx + buy_days_after + sell_days_after < len(df):
            open_price = df.iloc[idx + buy_days_after]['Open']
            close_price = df.iloc[idx + buy_days_after + sell_days_after]['Close']
            
            # 計算每次交易的報酬率
            if direction == 'short':
                trade_return = (open_price - close_price) / open_price  # 做空
            else:
                trade_return = (close_price - open_price) / open_price  # 做多
            
            trade_returns.append(trade_return)
            
            # 儲存每次交易的日期和報酬
            results['Date'].append(entry_date)
            results['Return'].append(trade_return)
    
    # 計算每次交易的平均報酬
    avg_return = np.sum(trade_returns) / len(trade_returns) if len(trade_returns) > 0 else np.nan
    
    # 計算總報酬
    total_return = np.prod([1 + r for r in trade_returns]) - 1 if len(trade_returns) > 0 else np.nan
    
    # 計算Sharpe Ratio（假設無風險利率為0）
    std_return = np.std(trade_returns) if len(trade_returns) > 0 else np.nan
    sharpe_ratio = avg_return / std_return if std_return != 0 else np.nan
    
    # 計算整體報酬率的勝率
    win_rate = round((np.array(trade_returns) > 0).sum() / len(trade_returns), 2) if len(trade_returns) > 0 else np.nan
    
    # 計算所有交易的最大回撤
    returns_series = pd.Series(trade_returns)
    max_dd = min(returns_series) if len(returns_series) > 0 else np.nan
    
    # 結果儲存
    summary = {
        'Average_Return': round(avg_return, 4),
        'Total_Return': round(total_return, 4),
        'Sharpe': round(sharpe_ratio, 4),
        'Win_Rate': round(win_rate, 4),
        'MaxDD': round(max_dd, 4)
    }
    
    return results, summary

def find_best_buy_sell_days_with_metrics(df, signal, direction, max_days):
    best_total_return = -np.inf  # 初始化為最小值
    best_buy_days = None
    best_sell_days = None
    best_trade_returns = []
    best_sharpe = None
    best_win_rate = None
    best_max_dd = None

    # 遍歷所有可能的買入和賣出天數組合
    for buy_days in range(1, max_days + 1):
        for sell_days in range(buy_days + 1, max_days + 1):  # 確保 sell_days 大於 buy_days
            results, summary = calc_metrics_for_signal(df, signal, direction, buy_days, sell_days)
            
            total_return = summary['Total_Return']
            trade_returns = results['Return']
            
            # 找出最大的 Total Return 及其對應的買入和賣出天數
            if total_return > best_total_return:
                best_total_return = total_return
                best_buy_days = buy_days
                best_sell_days = sell_days
                best_trade_returns = trade_returns
                best_sharpe = summary['Sharpe']  # 更新 Sharpe Ratio
                best_win_rate = summary['Win_Rate']  # 更新 Win Rate
                best_max_dd = summary['MaxDD']  # 更新最大回撤 MaxDD
    
    # 計算平均報酬率
    avg_return = np.mean(best_trade_returns) if len(best_trade_returns) > 0 else np.nan
    
    return best_buy_days, best_sell_days, best_total_return, avg_return, len(best_trade_returns), best_sharpe, best_win_rate, best_max_dd

# 設定最大天數範圍，例如最多等待60天內的買入與賣出
max_days = 70

start_time = time.time()

# 尋找做空策略下最佳買入和賣出天數，並計算平均報酬率、Sharpe、Win Rate 和 MaxDD
best_buy_days_raise, best_sell_days_raise, best_total_return_raise, avg_return_raise, num_trades_raise, sharpe_raise, win_rate_raise, max_dd_raise = find_best_buy_sell_days_with_metrics(df, 'Raise', 'short', max_days)
print(f"Best Raise Signal (Short) - Buy {best_buy_days_raise} days after, Sell {best_sell_days_raise} days after")
print(f"Total Return: {round(best_total_return_raise, 4)}, Average Return: {avg_return_raise}, Number of Trades: {num_trades_raise}")
print(f"Sharpe Ratio: {sharpe_raise}, Win Rate: {win_rate_raise}, Max Drawdown: {max_dd_raise}")
print("=" * 100)

# 尋找做多策略下最佳買入和賣出天數，並計算平均報酬率、Sharpe、Win Rate 和 MaxDD
best_buy_days_cut, best_sell_days_cut, best_total_return_cut, avg_return_cut, num_trades_cut, sharpe_cut, win_rate_cut, max_dd_cut = find_best_buy_sell_days_with_metrics(df, 'Cut', 'long', max_days)
print(f"Best Cut Signal (Long) - Buy {best_buy_days_cut} days after, Sell {best_sell_days_cut} days after")
print(f"Total Return: {round(best_total_return_cut, 4)}, Average Return: {avg_return_cut}, Number of Trades: {num_trades_cut}")
print(f"Sharpe Ratio: {sharpe_cut}, Win Rate: {win_rate_cut}, Max Drawdown: {max_dd_cut}")
print("=" * 100)

end_time = time.time()
print(f"Execution Time: {round(end_time - start_time, 4)} seconds")