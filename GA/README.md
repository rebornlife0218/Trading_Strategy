### 暑期 TMBA  
* 標的：BTCUSDT
* 使用方法：基因演算法
* 策略：

### 基因演算法(Genetic Algorithm, GA)
> 基因演算法是一種優化演算法，主要依據達爾文“進化論”中所提出的「物競天擇，適者生存，不適者淘汰」的生物演化法則而來，其核心觀念為模擬基因的優勝劣汰，以進行最佳化的計算。
* 名詞介紹：
1. 種群 (Population)\
可以想像成一個區域內的所有個體，數個個體集合起來就是種群

2. 染色體 (Chromosome)\
一個種群內有很多個體，在每個個體中有獨一無二的染色體

3. 基因 (Gene)\
每個染色體內的最小單位，不同的基因組合形成一個新的染色個體

4. 交配(Crossover)\
新的個體會遺傳父母雙方各一部分的基因，至於怎麼遺傳就是看你的 Crossover 方法。

5. 突變(Mutation)\
在遺傳交配的過程中同時有一定的概率發生基因突變，突變的用意很簡單，若一個種群中找到一個解但這個解只是區域最佳解(Local Optima)話再怎麼交配最後跑出來的值可能還是區域最佳解，此時若在交配過程中有一定的機率突變可能因此跨過障礙找到一個最佳的全域最佳解(Global Optimum)。

* 步驟說明\
Step 1：產生初始群集（Initial Population）\
Step 2：計算適應度 （Fitness）\
Step 3：複製（Reproduction）或選取（Selection）\
以下列出常見的兩種選擇方法
1. 競賽選擇(Tournament selection)
   
![image](https://github.com/user-attachments/assets/57752468-180d-48b8-8725-355ea6fdcfa0)

2.輪盤選擇(Roulette Wheel Selection)

![image](https://github.com/user-attachments/assets/18819f08-784b-410e-bc58-d8e5e4021d01)

Step 4：交配（Crossover）\
Step 5：突變（Mutation）\
Step 6：若達到 iteration 次數的上限時，則停止並輸出最佳解，若尚未滿足則回到Step 3。

***
### Reference
[Python — 基因演算法(Genetic Algorithm, GA)求解最佳化問題](https://medium.com/hunter-cheng/python-%E5%9F%BA%E5%9B%A0%E6%BC%94%E7%AE%97%E6%B3%95-genetic-algorithm-ga-%E6%B1%82%E8%A7%A3%E6%9C%80%E4%BD%B3%E5%8C%96%E5%95%8F%E9%A1%8C-b7e6d635922)
