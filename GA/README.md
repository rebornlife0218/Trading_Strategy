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
1. 競賽選擇(Tournament selection)\
   會從所有個體中隨機選出兩個，接著比較何者的解為佳，最後選擇較佳者進行交配。
![353688566-57752468-180d-48b8-8725-355ea6fdcfa0](https://github.com/user-attachments/assets/5dfc20d1-5988-40d4-a97a-d35a7259d836)


2. 輪盤選擇(Roulette Wheel Selection)\
   為一種回放式隨機取樣法，將一輪盤分成N個部分，根據fitness決定其盤面面積大小，fitness越大面積就越大，故在隨機取樣中被選到的機會就會越大。
![image](https://github.com/user-attachments/assets/04f6da45-dfd6-44b5-abb7-59c8b3aaca8f)


Step 4：交配（Crossover）\
Step 5：突變（Mutation）\
Step 6：若達到 iteration 次數的上限時，則停止並輸出最佳解，若尚未滿足則回到Step 3。

***
### Reference
[基因遺傳演算法（GA)](https://tzuchieh0931.medium.com/ga-metaheuristic-05-cf98c543da7f)
