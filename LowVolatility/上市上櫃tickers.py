import requests
from bs4 import BeautifulSoup
import pandas as pd
import re

def extract_ticker1(text):
    match = re.search(r'\d+', text)  # 提取數字部分
    if match:
        return f"{match.group()}.TW"
    return None

def extract_ticker2(text):
    match = re.search(r'\d+', text)  # 提取數字部分
    if match:
        return f"{match.group()}.TWO"
    return None

# 抓取第一個網頁資料
url1 = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=2"
res1 = requests.get(url1)
soup1 = BeautifulSoup(res1.text, "lxml")
rows1 = soup1.find_all('tr')[2:1003]

tds1 = []
for row in rows1:
    tds_list = row.find_all('td')
    if tds_list:
        ticker_text = tds_list[0].get_text().strip()
        ticker = extract_ticker1(ticker_text)
        if ticker:
            tds1.append([ticker])

# 抓取第二個網頁資料
url2 = "https://isin.twse.com.tw/isin/C_public.jsp?strMode=4"
res2 = requests.get(url2)
soup2 = BeautifulSoup(res2.text, "lxml")
rows2 = soup2.find_all('tr')

tds2 = []
for row in rows2:
    tds_list = row.find_all('td')
    if len(tds_list) >= 6:  # 確保至少有六個 <td>
        sixth_td = tds_list[5]  # 取得第六個 <td>
        if sixth_td.get_text().strip() == 'ESVUFR':
            ticker_text = tds_list[0].get_text().strip()
            ticker = extract_ticker2(ticker_text)
            if ticker:
                tds2.append([ticker])

tds_combined = tds1 + tds2
df = pd.DataFrame(tds_combined, columns=["ticker"])
df.to_csv('上市上櫃tickers.txt', index=False, header=False, sep='\t')
print("Data saved to 上市上櫃tickers.txt")
