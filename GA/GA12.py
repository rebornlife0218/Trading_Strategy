import pandas as pd
import numpy as np
from deap import base, creator, tools, algorithms
import random
import matplotlib.pyplot as plt

# 讀取數據
data = pd.read_csv('BTCUSDT_1d.csv')
data['date'] = pd.to_datetime(data['date'])
close_prices = data.set_index('date')['close']

# 設定樣本內和樣本外期間
sample_in_start = '2020-01-01'
sample_in_end = '2022-08-31'
sample_out_start = '2022-09-01'
sample_out_end = '2023-06-30'

sample_in = close_prices.loc[sample_in_start:sample_in_end]
sample_out = close_prices.loc[sample_out_start:sample_out_end]

random.seed(20240719)

# 回測函數
def backtest(very_short_window, short_window, long_window, very_long_window, prices):
    if (not isinstance(very_short_window, int) or not isinstance(short_window, int) or      # 判斷一個變量是否屬於某個類型
        not isinstance(long_window, int) or not isinstance(very_long_window, int) or 
        very_short_window <= 0 or short_window <= 0 or long_window <= 0 or very_long_window <= 0):
        return -np.inf, []

    if not (very_long_window > long_window > short_window > very_short_window):
        return -np.inf, []
    
    # 初始化資金
    initial_capital = 10000
    available_capital = initial_capital
    trade_unit_value = 4990  # 固定的交易單位價值

    # 計算移動平均
    very_short_ma = prices.rolling(window=very_short_window).mean()
    short_ma = prices.rolling(window=short_window).mean()
    long_ma = prices.rolling(window=long_window).mean()
    very_long_ma = prices.rolling(window=very_long_window).mean()

    # 填充 NaN 值
    very_short_ma.fillna(method='ffill', inplace=True)
    short_ma.fillna(method='ffill', inplace=True)
    long_ma.fillna(method='ffill', inplace=True)
    very_long_ma.fillna(method='ffill', inplace=True)

    position = np.zeros(len(prices), dtype=np.float32)      # 用float32 可以節省內存並提高計算效率
    trades = []  # 儲存交易紀錄
    units = 0  # 持有單位數量
    fee = 0.0004  # 0.04%

    for i in range(max(very_short_window, short_window, long_window, very_long_window), len(prices)):
        current_price = prices.iloc[i]
        current_date = prices.index[i]

        # 買入
        if units == 0:
            if (very_short_ma.iloc[i] > short_ma.iloc[i] > long_ma.iloc[i] > very_long_ma.iloc[i] and
                very_short_ma.iloc[i-1] > short_ma.iloc[i-1] > long_ma.iloc[i-1] > very_long_ma.iloc[i-1]):
                    
                if available_capital >= trade_unit_value:  # 檢查資金是否足夠
                    unit_size = trade_unit_value / current_price
                    available_capital -= unit_size * current_price * (1 + fee)  # 更新可用資金
                    position[i] = unit_size
                    units += unit_size
                    trades.append(f"Buy {unit_size:.2f} units on {current_date.strftime('%Y-%m-%d')} at price {current_price:.2f}")

        # 已持倉時的賣出、停損、停利
        if units > 0:
            if (very_short_ma.iloc[i] < short_ma.iloc[i] < long_ma.iloc[i] < very_long_ma.iloc[i] and
                very_short_ma.iloc[i-1] < short_ma.iloc[i-1] < long_ma.iloc[i-1] < very_long_ma.iloc[i-1]):

                    unit_size = trade_unit_value / current_price  # 計算要賣出的單位數量
                    available_capital += unit_size * current_price * (1 - fee)  # 更新可用資金
                    position[i] = -unit_size
                    profit_loss = unit_size * (current_price - position[i-1] if i > 0 else 0)     # 確保在 i = 0 的時候，不會發生無效的索引操作（即 position[-1]）
                    trades.append(f"Sell {unit_size:.2f} units on {current_date.strftime('%Y-%m-%d')} at price {current_price:.2f}")
                    units -= unit_size  # 更新持有單位數量

            elif (very_short_ma.iloc[i] > short_ma.iloc[i] > long_ma.iloc[i] and 
                prices.iloc[i] < very_short_ma.iloc[i] and prices.iloc[i] < short_ma.iloc[i] and
                prices.iloc[i] < long_ma.iloc[i]):

                profit_loss = units * (current_price - position[i-1] if i > 0 else 0)     # 確保在 i = 0 的時候，不會發生無效的索引操作（即 position[-1]）
                available_capital += units * current_price * (1 - fee)  # 更新可用資金
                position[i] = -units
                trades.append(f"Stop loss {units:.2f} units on {current_date.strftime('%Y-%m-%d')} at price {current_price:.2f}")
                units = 0

            elif (very_short_ma.iloc[i] > short_ma.iloc[i] > long_ma.iloc[i]  and
                very_short_ma.iloc[i-1] > short_ma.iloc[i-1] > long_ma.iloc[i-1]  and
                prices.iloc[i] > prices.iloc[i-1] > prices.iloc[i-2] > prices.iloc[i-3] > prices.iloc[i-4]):

                unit_size = trade_unit_value / current_price  # 計算要賣出的單位數量
                available_capital += unit_size * current_price * (1 - fee)  # 更新可用資金
                profit_loss = unit_size * (current_price - position[i-1] if i > 0 else 0)  # 計算損益
                position[i] = -unit_size
                trades.append(f"Take profit {unit_size:.2f} units on {current_date.strftime('%Y-%m-%d')} at price {current_price:.2f}")
                units -= unit_size  # 更新持有單位數量

            # 加倉
            elif (very_short_ma.iloc[i] > short_ma.iloc[i] > long_ma.iloc[i] > very_long_ma.iloc[i] and
                very_short_ma.iloc[i-1] > short_ma.iloc[i-1] > long_ma.iloc[i-1] > very_long_ma.iloc[i-1] and
                very_short_ma.iloc[i-2] > short_ma.iloc[i-2] > long_ma.iloc[i-2] > very_long_ma.iloc[i-2] and
                prices.iloc[i] > prices.iloc[i-1] > prices.iloc[i-2]):

                if available_capital >= trade_unit_value:  # 檢查資金是否足夠
                    unit_size = trade_unit_value / current_price
                    available_capital -= unit_size * current_price * (1 + fee)  # 更新可用資金
                    position[i] = unit_size
                    units += unit_size
                    trades.append(f"Overweight {unit_size:.2f} units on {current_date.strftime('%Y-%m-%d')} at price {current_price:.2f}")

    # 計算累積報酬率
    returns = prices.pct_change().shift(-1).dropna()
    cumulative_returns = (1 + returns).cumprod() - 1

    # 確保 position 和 cumulative_returns 長度相同
    min_length = min(len(position), len(cumulative_returns))
    position = position[:min_length]
    cumulative_returns = cumulative_returns[:min_length]

    # 計算策略報酬
    strategy_returns = position * cumulative_returns
    strategy_returns.dropna(inplace=True)
    if strategy_returns.empty:
        return -np.inf, []

    # 計算年化報酬率
    total_return = (1 + strategy_returns.sum()) - 1
    if total_return <= -1:
        annualized_return = float('-inf')
    else:
        annualized_return = ((1 + total_return) ** (365 / len(strategy_returns))) - 1

    return float(annualized_return), trades

# 定義適應度函數
def eval_strategy(individual):
    very_short_window, short_window, long_window, very_long_window = map(int, individual)
    annualized_return, _ = backtest(very_short_window, short_window, long_window, very_long_window, sample_in)
    return (annualized_return,)     # 明確地定義了返回一個元組

# 設置基因演算法
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("attr_vs", random.randint, 1, 200)
toolbox.register("attr_s", random.randint, 1, 200)
toolbox.register("attr_l", random.randint, 1, 200)
toolbox.register("attr_vl", random.randint, 1, 200)

def valid_individual():
    while True:
        very_short_window = toolbox.attr_vs()
        short_window = toolbox.attr_s()
        long_window = toolbox.attr_l()
        very_long_window = toolbox.attr_vl()
        if very_short_window < short_window < long_window < very_long_window < 200:
            return [very_short_window, short_window, long_window, very_long_window]
        else:
            return valid_individual()  # 重新生成符合條件的個體

toolbox.register("individual", tools.initIterate, creator.Individual, valid_individual)     
toolbox.register("population", tools.initRepeat, list, toolbox.individual)      
toolbox.register("mate", tools.cxBlend, alpha=0.8)
toolbox.register("mutate", tools.mutUniformInt, low=[1, 1, 1, 1], up=[200, 200, 200, 200], indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=4)
toolbox.register("evaluate", eval_strategy)

# 初始化族群
population = toolbox.population(n=200)
ngen = 30
cxpb = 0.85
mutpb = 0.15

# 設置統計數據
stats = tools.Statistics(lambda ind: ind.fitness.values[0] if ind.fitness.values else float('-inf'))
stats.register("avg", np.mean)
stats.register("max", np.max)

logbook = tools.Logbook()
logbook.header = ['gen', 'nevals'] + stats.fields + ['best_annualized_return']

# 收斂檢查設定
convergence_threshold = 1e-6
convergence_generations = 8
converged = False
converged_count = 0

# GA 主程式
for gen in range(ngen):
    offspring = toolbox.select(population, len(population))
    offspring = list(map(toolbox.clone, offspring))

    # 交叉
    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < cxpb:
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values

    # 變異
    for mutant in offspring:
        if random.random() < mutpb:
            toolbox.mutate(mutant)
            del mutant.fitness.values

    # 計算適應度
    invalid_ind = [ind for ind in offspring if not ind.fitness.valid]       # 找出適應度尚未計算的後代個體
    fitnesses = map(toolbox.evaluate, invalid_ind)      # 計算所有尚未計算適應度的個體的適應度
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    population[:] = toolbox.select(population + offspring, len(population))     # 選擇下一代種群

    record = stats.compile(population)
    best_ind = tools.selBest(population, 1)[0]
    best_annualized_return = best_ind.fitness.values[0]
    record['best_annualized_return'] = best_annualized_return
    logbook.record(gen=gen, nevals=len(invalid_ind), **record)      # 展開 record 字典中的所有鍵值對，將統計數據添加到記錄中。
    print(logbook.stream)

    # 檢查收斂
    if gen > 0:
        previous_best = logbook.select('best_annualized_return')[-2]
        if abs(previous_best - best_annualized_return) < convergence_threshold:
            converged_count += 1
        else:
            converged_count = 0
        if converged_count >= convergence_generations:
            converged = True
            print(f"GA converged at generation {gen}")
            break

if not converged:
    print("GA did not converge within the given number of generations.")

# 繪製收斂過程
gen = logbook.select('gen')
fit_maxs = logbook.select('max')
fit_avgs = logbook.select('avg')

plt.figure()
plt.plot(gen, fit_maxs, label='Max Fitness')
plt.plot(gen, fit_avgs, label='Avg Fitness')
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('GA Convergence')
plt.legend()
plt.grid()
plt.show()

# 獲得最佳個體
best_individual = tools.selBest(population, k=1)[0]
very_short_window, short_window, long_window, very_long_window = map(int, best_individual)
annualized_return_in, trades_in = backtest(very_short_window, short_window, long_window, very_long_window, sample_in)
annualized_return_out, trades_out = backtest(very_short_window, short_window, long_window, very_long_window, sample_out)

# 顯示樣本內的交易紀錄
for trade in trades_in:
    print(trade)
print("=" * 80)

# 顯示樣本外的交易紀錄
for trade in trades_out:
    print(trade)
print("=" * 80)

print(f"Best Individual: Very Short Window: {very_short_window}, Short Window: {short_window}, Long Window: {long_window}, Very Long Window: {very_long_window}")
print(f"Annualized Return (In-sample): {annualized_return_in}")
print(f"Annualized Return (Out-of-sample): {annualized_return_out}")

# 繪製回測結果
plt.figure(figsize=(12, 6))
plt.plot(sample_in.index, sample_in.values, label='In-sample Prices')
plt.plot(sample_out.index, sample_out.values, label='Out-of-sample Prices')
split_point = sample_in.index[-1]  # 樣本內的最後一個時間點
plt.axvline(x=split_point, color='red', linestyle='--', label='Split Point')
plt.title('BTCUSDT In-sample and Out-of-sample Prices')
plt.legend()
plt.show()
