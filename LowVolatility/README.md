# 低波動因子報酬 vs. 台股加權指數報酬

先抓取台灣上市上櫃ticker，網址包含權證需要過濾掉
> 也可以直接在yahoo finance抓，此作法只是方便以後會使用到上市上櫃ticker。
> 
[上市ticker](https://isin.twse.com.tw/isin/C_public.jsp?strMode=2)
[上櫃ticker](https://isin.twse.com.tw/isin/C_public.jsp?strMode=4)

選取近十年(start_date = "2014-06-30" end_date = "2024-06-30")波動率最低的10檔股票，各按等權重10%組成portfolio，與台股加權指數進行比較。(省略手續費、交易稅，也未考慮胃納量)
1. 讀取股票代號並下載歷史數據。
2. 根據波動度選擇每月最小波動度的股票，並構建等權重的投資組合。
3. 回測策略，計算每日報酬和累積報酬。
4. 計算策略和指數的關鍵績效指標，如：Sharpe Ratio、最大回撤、年化報酬率和勝率。
5. 繪製策略和指數的報酬及回撤圖，以便進行比較分析。

![LowVolatilityEffect_TW_result](https://github.com/user-attachments/assets/d216d309-8d0f-41fe-8a30-3733ace04821)
