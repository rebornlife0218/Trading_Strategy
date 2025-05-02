import pandas as pd
import numpy as np

# 讀取資料
file_path = 'TLT_23years.feather'
data = pd.read_feather(file_path)
data['Date'] = pd.to_datetime(data['Date'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month
df = pd.DataFrame(data)
df['Return'] = df['Close'].pct_change()

# 選舉日為台灣時間，但當日收盤後才知，故加一天(相當於隔日開盤已知)，但每年開票時間不確定。
election_dates = pd.to_datetime(['2024-11-06', '2020-11-04', '2016-11-09', '2012-11-07', '2008-11-05', '2004-11-03', '2000-11-08'])
election_dates = election_dates + pd.DateOffset(days=1)
election_returns = {}
d = 10

# 找出每個選舉日的前d天和後d天的持有期間報酬
for date in election_dates:
    # 選舉日的index
    idx = df.index[df['Date'] == date]
    if not idx.empty:
        idx = idx[0]
        
        # 前d天持有期間
        start_pre = max(0, idx - d)
        if idx - start_pre < d:  # 若不滿d天，延伸範圍
            start_pre = 0
        buy_pre_date = df.loc[start_pre, 'Date'].strftime('%Y-%m-%d')
        buy_pre_price = round(df.loc[start_pre, 'Open'], 2)
        
        sell_pre_date = df.loc[idx - 1, 'Date'].strftime('%Y-%m-%d')
        sell_pre_price = round(df.loc[idx - 1, 'Close'], 2)
        pre_days_return = (sell_pre_price / buy_pre_price) - 1  # 前d天的持有報酬
        
        # 後d天持有期間
        end_post = min(len(df) - 1, idx + d)
        if end_post - idx < d:  # 若不滿d天，延伸範圍
            end_post = len(df) - 1
        buy_post_date = df.loc[idx, 'Date'].strftime('%Y-%m-%d')
        buy_post_price = round(df.loc[idx, 'Open'], 2)
        
        sell_post_date = df.loc[end_post, 'Date'].strftime('%Y-%m-%d')
        sell_post_price = round(df.loc[end_post, 'Close'], 2)
        post_days_return = (sell_post_price / buy_post_price) - 1  # 後d天的持有報酬
        
        # 儲存到字典
        election_returns[date.strftime('%Y-%m-%d')] = {
            'Pre_days': {
                'Buy Date': buy_pre_date,
                'Buy Price': buy_pre_price,
                'Sell Date': sell_pre_date,
                'Sell Price': sell_pre_price,
                'Return': pre_days_return
            },
            'Post_days': {
                'Buy Date': buy_post_date,
                'Buy Price': buy_post_price,
                'Sell Date': sell_post_date,
                'Sell Price': sell_post_price,
                'Return': post_days_return
            }
        }

# 印出結果
for date, data in election_returns.items():
    print(f"Election Date: {date}")
    print(f"Pre {d} Days - Buy Date: {data['Pre_days']['Buy Date']}, "
          f"Buy Price: {data['Pre_days']['Buy Price']}, "
          f"Sell Date: {data['Pre_days']['Sell Date']}, "
          f"Sell Price: {data['Pre_days']['Sell Price']}, "
          f"Return: {data['Pre_days']['Return']:.2%}")
    
    print(f"Post {d} Days - Buy Date: {data['Post_days']['Buy Date']}, "
          f"Buy Price: {data['Post_days']['Buy Price']}, "
          f"Sell Date: {data['Post_days']['Sell Date']}, "
          f"Sell Price: {data['Post_days']['Sell Price']}, "
          f"Return: {data['Post_days']['Return']:.2%}")
    print("=" * 120)

pre_election_returns = [data['Pre_days']['Return'] for data in election_returns.values()]
post_election_returns = [data['Post_days']['Return'] for data in election_returns.values()]

def calculate_metrics(returns):
    total_return = np.prod([1 + r for r in returns]) - 1 
    avg_return = np.mean(returns)  
    sharpe_ratio = avg_return / np.std(returns) if np.std(returns) != 0 else 0  
    win_rate = sum(1 for r in returns if r > 0) / len(returns)  
    max_dd = min(returns)  # Max Drawdown (直接取最大負報酬)
    return {
        'Total Return': total_return,
        'Average Return': avg_return,
        'Sharpe Ratio': sharpe_ratio,
        'Win Rate': win_rate,
        'Max Drawdown': max_dd
    }

pre_election_metrics = calculate_metrics(pre_election_returns)
post_election_metrics = calculate_metrics(post_election_returns)

print("Pre election period:")
for key, value in pre_election_metrics.items():
    print(f"{key}: {value:.2%}" if isinstance(value, float) else f"{key}: {value}")

print("\nPost election period:")
for key, value in post_election_metrics.items():
    print(f"{key}: {value:.2%}" if isinstance(value, float) else f"{key}: {value}")