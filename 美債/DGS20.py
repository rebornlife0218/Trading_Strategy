import pandas as pd
import os

# 讀取資料
file_path = 'DGS20.csv'
data = pd.read_csv(file_path)
data['Date'] = pd.to_datetime(data['DATE'])
data['Year'] = data['Date'].dt.year
data['Month'] = data['Date'].dt.month

# 創建 DataFrame
df = pd.DataFrame(data)
df = df[df['DGS20'] != '.']     # 刪除 DGS20 為 '.' 的行

# 將 DGS20 轉換為浮點數
df['DGS20'] = df['DGS20'].astype(float)


print(df[['DATE', 'DGS20']])