import pandas as pd
import numpy as np

# 讀取資料
file_path = 'TLT_23years.feather'
data = pd.read_feather(file_path)
data['Date'] = pd.to_datetime(data['Date'])
df = pd.DataFrame(data)
df = df[df['Date'].dt.year != 2025]     # 2025/11 無資料

# 儲存每年的交易結果
annual_trades = {}
all_returns = []  # 用於計算總報酬等指標

def find_date(df, target_date, find_forward=False):
    if target_date in df['Date'].values:
        return target_date
    if find_forward:
        # 向後查找
        closest_date_idx = df['Date'].searchsorted(target_date, side='right') - 1
    else:
        # 向前查找
        closest_date_idx = df['Date'].searchsorted(target_date, side='left')
    if 0 <= closest_date_idx < len(df):
        return df['Date'].iloc[closest_date_idx]
    return None

# 計算報酬的函數
def calculate_trade_data(df, entry_date_target, exit_date_target):
    # 計算交易的開盤和收盤資料，並確保使用最近的可用日期
    if entry_date_target == '11/11':
        # 如果是 11/11 開盤，強制向後查找
        entry_date = find_date(df, entry_date_target, find_forward=True)
    else:
        entry_date = find_date(df, entry_date_target, find_forward=False)
    
    exit_date = find_date(df, exit_date_target, find_forward=True)  # 始終向後查找平倉日
    
    if entry_date is not None and exit_date is not None:
        entry_price = df.loc[df['Date'] == entry_date, 'Open'].values[0]
        exit_price = df.loc[df['Date'] == exit_date, 'Close'].values[0]
        return entry_date, entry_price, exit_date, exit_price
    return None, None, None, None

# 遍歷每年，計算交易的開盤和收盤資料
for year in df['Date'].dt.year.unique():
    # 篩選該年的數據
    year_data = df[df['Date'].dt.year == year]
    
    # 計算交易 11/1 開盤買，11/10 收盤賣
    entry_date_1, entry_price_1, exit_date_1, exit_price_1 = calculate_trade_data(year_data, f'{year}-11-01', f'{year}-11-10')
    
    # 計算交易 11/11 開盤買，11/20 收盤賣
    entry_date_2, entry_price_2, exit_date_2, exit_price_2 = calculate_trade_data(year_data, f'{year}-11-11', f'{year}-11-20')

    # 計算漲跌幅
    return_1 = (exit_price_1 - entry_price_1) / entry_price_1 if entry_price_1 else np.nan
    return_2 = (exit_price_2 - entry_price_2) / entry_price_2 if entry_price_2 else np.nan

    # 儲存結果
    annual_trades[year] = {
        'Trade_1': {
            'Entry_Date': entry_date_1,
            'Entry_Price': entry_price_1,
            'Exit_Date': exit_date_1,
            'Exit_Price': exit_price_1,
            'Return': return_1
        },
        'Trade_2': {
            'Entry_Date': entry_date_2,
            'Entry_Price': entry_price_2,
            'Exit_Date': exit_date_2,
            'Exit_Price': exit_price_2,
            'Return': return_2
        }
    }
    
    # 收集所有交易的報酬以便計算整體指標
    if return_1 is not None:
        all_returns.append(return_1)
    if return_2 is not None:
        all_returns.append(return_2)

# 印出每年的交易資料和漲跌幅
for year, trades in annual_trades.items():
    for trade_name, trade_info in trades.items():
        if trade_info['Entry_Date'] is not None and trade_info['Exit_Date'] is not None:
            print(f"Year: {year} - Buy Date: {trade_info['Entry_Date'].date()} - Buy Price: {trade_info['Entry_Price']:.2f} "
                  f"- Sell Date: {trade_info['Exit_Date'].date()} - Sell Price: {trade_info['Exit_Price']:.2f} "
                  f"- Return: {trade_info['Return']:.2%}")  # 印出漲跌幅
        else:
            print(f"  {trade_name} has no valid trading data.")
    print("=" * 120)  

# 提取 Trade_2 的回報
trade_2_returns = []

for year, trades in annual_trades.items():
    if 'Trade_2' in trades and trades['Trade_2']['Return'] is not None:
        return_value = trades['Trade_2']['Return']
        if not np.isnan(return_value):  # 過濾 nan 值
            trade_2_returns.append(return_value)

# 總報酬
total_return = sum(trade_2_returns)

# 平均報酬
average_return = np.mean(trade_2_returns) if trade_2_returns else 0

# 夏普比率（假設無風險利率為0）
risk_free_rate = 0
sharpe_ratio = (average_return - risk_free_rate) / np.std(trade_2_returns) if np.std(trade_2_returns) > 0 else np.nan

# 勝率
win_count = sum(1 for r in trade_2_returns if r > 0)
total_trades = len(trade_2_returns)
win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0

# 計算最大回撤
drawdown = min([r for r in trade_2_returns if r < 0], default=0)

print('11月中:')
print(f"Total Return : {total_return:.2%}")
print(f"Average Return : {average_return:.2%}")
print(f"Sharpe Ratio : {sharpe_ratio:.4f}")
print(f"Win Rate : {win_rate:.2f}%")
print(f"Max Drawdown : {drawdown:.2%}")