import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from tabulate import tabulate

# 讀取CSV檔案，將非數字和空格替換為 NaN
on_rrp = pd.read_csv('on_rrp_0729.csv', parse_dates=[0], index_col=0, na_values=[' ', 'non_numeric_value'])
SP500 = pd.read_csv('S&P500_sorted.csv', parse_dates=[0], index_col=0, na_values=[' ', 'non_numeric_value'])

# 將數據轉換為數字型態
on_rrp['ON RRP'] = pd.to_numeric(on_rrp['ON RRP'], errors='coerce')
SP500['Close'] = pd.to_numeric(SP500['Close'], errors='coerce')
data = pd.merge(on_rrp, SP500[['Close']].round(1), left_index=True, right_index=True)

# 遺失值處理
on_rrp['ON RRP'] = on_rrp['ON RRP'].fillna(method='ffill')
SP500['Close'] = SP500['Close'].fillna(method='ffill')

def calculate_on_rrp_change(data):
    data['on_rrp_change'] = data['ON RRP'].diff() / data['ON RRP'].shift(1)
    return data['on_rrp_change']

data['on_rrp_change'] = calculate_on_rrp_change(data)

def calculate_ma(data, column, window):
    return data[column].rolling(window=window).mean()

data['on_rrp_10ma'] = calculate_ma(data, 'ON RRP', window=10)

def calculate_position(data, column, ma_column, shift_days, unit=1):
    data['inv'] = 0  # 初始化庫存為 0

    for i in range(1, len(data)):
        if data[column].iloc[i-1] < data[ma_column].shift(shift_days).iloc[i-1]:
            data.loc[data.index[i], 'inv'] = data['inv'].iloc[i-1] + 1
        elif data[column].iloc[i-1] > data[ma_column].shift(shift_days).iloc[i-1] and data['inv'].iloc[i-1] >= 1:
            data.loc[data.index[i], 'inv'] = data['inv'].iloc[i-1] - 1
        else:
            data.loc[data.index[i], 'inv'] = data['inv'].iloc[i-1]

    condition_buy = (data[column] < data[ma_column].shift(shift_days))
    condition_sell = (data[column] > data[ma_column].shift(shift_days)) & (data['inv'] >= 1)

    buy_signals = np.where(condition_buy, 1, 0)
    sell_signals = np.where(condition_sell, -1, 0)
    
    return buy_signals + sell_signals

data['position'] = calculate_position(data, 'ON RRP', 'on_rrp_10ma', shift_days=3)

# 計算每日報酬率
data['daily_return'] = data['position'].shift(1) * data['Close'].pct_change()

# 計算總體報酬率
total_return = (1 + data['daily_return']).cumprod() - 1

# 確保 non_nan_returns 中有數據
non_nan_returns = data['daily_return'].dropna()

# 計算 Annualized Return
annualized_return = (1 + non_nan_returns).cumprod().iloc[-1] ** (252 / len(non_nan_returns)) - 1
    
# 計算 Sharpe ratio
risk_free_rate = 0
sharpe_ratio = (non_nan_returns.mean() - risk_free_rate) / non_nan_returns.std() * np.sqrt(252)

# 計算 Sortino ratio
risk_free_rate = 0
downside_risk = np.sqrt((np.minimum(non_nan_returns - risk_free_rate, 0) ** 2).sum() / len(non_nan_returns))
sortino_ratio = (non_nan_returns.mean() - risk_free_rate) / downside_risk * np.sqrt(252)

# 計算 Max Drawdown 
cumulative_return = (1 + non_nan_returns).cumprod()
drawdown = cumulative_return / cumulative_return.cummax() - 1
max_drawdown = drawdown.min()

# 整理結果
results = [
    ["Annualized Return", annualized_return],
    ["Sharpe Ratio", sharpe_ratio],
    ["Sortino Ratio", sortino_ratio],
    ["Max Drawdown", max_drawdown]
]

print(tabulate(results, headers=["Metric", "Value"], tablefmt="grid"))

# 印出進場和出場時點並印出
buy_signals = data[data['position'] == 1].index
sell_signals = data[data['position'] == -1].index
print("\nBuy Signals:")
print(buy_signals)
print("\nSell Signals:")
print(sell_signals)

# 繪製進場和出場點
plt.figure(figsize=(14, 7))
plt.plot(data.index, data['Close'], label='S&P500 Close', color='blue')
plt.scatter(buy_signals, data.loc[buy_signals]['Close'], marker='^', color='r', label='Buy Signal')
plt.scatter(sell_signals, data.loc[sell_signals]['Close'], marker='v', color='g', label='Sell Signal')
plt.title('S&P500 Trading Strategy with ON RRP Volume 10 MA')
plt.xlabel('Date')
plt.ylabel('Close Price')
plt.legend()

# 設置 x 軸格式為年月
ax = plt.gca()
ax.xaxis.set_major_locator(mdates.MonthLocator(bymonth=range(1, 13, 3)))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))

plt.xticks(rotation=45)
plt.show()
