import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tabulate import tabulate

# 讀取CSV檔案，將非數字和空格替換為 NaN
on_rrp = pd.read_csv('on_rrp.csv', parse_dates=[0], index_col=0, na_values=[' ', 'non_numeric_value'])  # parse_dates=[0] 是用來指定將 CSV 文件中的哪一列解析為日期，並將其作為DataFrame的索引，index_col=0 將解析後的日期列變成DataFrame的索引。
twii = pd.read_csv('twii.csv', parse_dates=[0], index_col=0, na_values=[' ', 'non_numeric_value'])

on_rrp['ON RRP'] = pd.to_numeric(on_rrp['ON RRP'], errors='coerce')     # pd.to_numeric 函數用於將一列的數據轉換為數字，同時 errors='coerce' 參數表示在轉換過程中遇到無法轉換的值時，將其替換為 NaN
twii['Adj Close'] = pd.to_numeric(twii['Adj Close'], errors='coerce')
data = pd.merge(on_rrp, twii[['Adj Close']].round(1), left_index=True, right_index=True)

# 遺失值處理
on_rrp['ON RRP'] = on_rrp['ON RRP'].fillna(method='ffill')
twii['Adj Close'] = twii['Adj Close'].fillna(method='ffill')


def calculate_on_rrp_change(data):      # 計算短期內ON RRP交易量的變化
    data['on_rrp_change'] = data['ON RRP'].diff() / data['ON RRP'].shift(1)
    return data['on_rrp_change']

data['on_rrp_change'] = calculate_on_rrp_change(data)     

def calculate_ma(data, column, window):     # 計算十日線
    return data[column].rolling(window=window).mean()

data['on_rrp_10ma'] = calculate_ma(data, 'ON RRP', window=10)     

# 策略：當前的ON RRP交易量小於其過去3天的10ma，就買入指數。當前的ON RRP交易量大於其過去3天的10ma，就賣出指數。
def calculate_position(data, column, ma_column, shift_days, unit=1):
    data['inv'] = 0  # 初始化庫存為 0

    for i in range(1, len(data)):       # 從第二行到最後一行
        if data[column].iloc[i-1] < data[ma_column].shift(shift_days).iloc[i-1]:
            # 買進
            data.loc[data.index[i], 'inv'] = data['inv'].iloc[i-1] + 1
        elif data[column].iloc[i-1] > data[ma_column].shift(shift_days).iloc[i-1] and data['inv'].iloc[i-1] >= 1:
            # 賣出（需擁有庫存）
            data.loc[data.index[i], 'inv'] = data['inv'].iloc[i-1] - 1
        else:
            # 保持不變
            data.loc[data.index[i], 'inv'] = data['inv'].iloc[i-1]

    condition_buy = (data[column] < data[ma_column].shift(shift_days))
    condition_sell = (data[column] > data[ma_column].shift(shift_days)) & (data['inv'] >= 1)

    buy_signals = np.where(condition_buy, 1, 0)
    sell_signals = np.where(condition_sell, -1, 0)
    
    return buy_signals + sell_signals

data['position'] = calculate_position(data, 'ON RRP', 'on_rrp_10ma', shift_days=3)

# 將前一天的持倉狀態乘以當天的Adj Close價格變化率，得到的結果就是每天的報酬率。
data['daily_return'] = data['position'].shift(1) * data['Adj Close'].pct_change()

# 計算總體報酬率
total_return = (1 + data['daily_return']).cumprod() - 1     # cumprod()：進行累積乘積，得到每天的累積報酬

# 計算 Annualized Return
non_nan_returns = data['daily_return'].dropna()
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
plt.plot(data.index, data['Adj Close'], label='TWII Adj Close', color='blue')
plt.scatter(buy_signals, data.loc[buy_signals]['Adj Close'], marker='^', color='r', label='Buy Signal')
plt.scatter(sell_signals, data.loc[sell_signals]['Adj Close'], marker='v', color='g', label='Sell Signal')
plt.title('TWII Trading Strategy with ON RRP Volume 10 MA')
plt.xlabel('Date')
plt.ylabel('Adj Close Price')
plt.legend()
plt.show()
