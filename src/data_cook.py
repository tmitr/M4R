import numpy as np
import pandas as pd
from datetime import datetime
import pyarrow.feather as feather
import gc

BIG_NUMBER = 1e25
SMALL_NUMBER = 1/BIG_NUMBER
POOL_FEE = .0005

directory = r'C:\data\test'


timestamps = feather.read_feather(directory+'\\'+'block_time_map').head(25000)
readable_timestamps = np.array([datetime.utcfromtimestamp(int(ts)) for ts in timestamps.values if int(ts)!=0])
timestamps = pd.Series(readable_timestamps, index=timestamps.index)

df_mint = feather.read_feather(directory+'\\'+'mint')
df_burn = feather.read_feather(directory+'\\'+'burn')
df_swap=feather.read_feather(directory+'\\'+'swap_aa')


df_swap['Type'] = 'swap'
df_mint['Type'] = 'mint'
df_burn['Type'] = 'burn'
df_burn.loc[:, 'liquidity'] *= -1

df = pd.concat([df_swap, df_mint, df_burn], ignore_index=True).sort_values(['blockNumber', 'logIndex']).reset_index(drop=True)
flag_ts=df['blockNumber']<=timestamps.index[-1]
df.loc[flag_ts,'timestamp'] = timestamps[df.loc[flag_ts,'blockNumber']].values

df=df.head(10000)
del df_swap, df_mint, df_burn
gc.collect()
df['price'].ffill(inplace=True)
df['rangeLow'].ffill(inplace=True)
df['rangeHigh'].ffill(inplace=True)
df.loc[df.Type == 'swap', 'liquidity'] = 0




dd1= df.pivot_table(index=df.index, columns=['rangeLow'], values='liquidity', fill_value=0).cumsum() #Group together liquidity with same rangeLow, c
dd2 = df.pivot_table(index=df.index, columns=['rangeHigh'], values='liquidity', fill_value=0).cumsum()
dd3 = dd1.subtract(dd2, fill_value=0) # Subtract range highs and lows to only get net liquidity delta between each tick value
del dd1, dd2
gc.collect()


histogram = dd3.cumsum(axis=1)
# Sum liquidity deltas to get actual amount of liquidity between each value
#histogram['Type']=df['Type']


levels = dd3.columns.to_numpy()[np.newaxis, :]
prices = df['price'].to_numpy()[:, np.newaxis]
df['pa'] = np.max(levels*(levels < prices)*(np.abs(dd3)>0), axis=1)
df['pb'] = np.min(levels + BIG_NUMBER*((np.abs(dd3)==0) + (levels<=prices)), axis=1)

prices_prev =  df['price'].shift(1).to_numpy()[:, np.newaxis]
# x tokem = ETH
# y token = USDC

direction = prices > prices_prev
x_fee_density = (np.sqrt(np.minimum(prices, levels[:,1:])) - np.sqrt(np.maximum(prices_prev, levels[:, :-1]))) * (prices > levels[:, :-1]) * (prices_prev < levels[:, 1:]) * direction
x_fee_density_available = (np.sqrt(np.minimum(prices, levels[:,1:])) - np.sqrt(np.maximum(0, levels[:, :-1]))) * (prices > levels[:, :-1]) * (0 < levels[:, 1:])

#x_fee_density_available = -(np.sqrt(np.minimum(0, levels[:,1:])) - np.sqrt(np.maximum(prices_prev, levels[:, :-1])))
#x_fee_density_available = (np.sqrt(np.minimum(BIG_NUMBER, levels[:,1:])) - np.sqrt(np.maximum(prices_prev, levels[:, :-1]))) * (BIG_NUMBER > levels[:, :-1]) * (prices_prev < levels[:, 1:])
y_fee_density = (np.sqrt(np.minimum(1/prices, 1/levels[:,:-1])) - np.sqrt(np.maximum(1/prices_prev, 1/levels[:, 1:]))) * (prices < levels[:, 1:]) * (prices_prev > levels[:, :-1]) * (~direction)
#y_fee_density_available = (np.sqrt(np.minimum(1/SMALL_NUMBER, 1/levels[:,:-1])) - np.sqrt(np.maximum(1/prices_prev, 1/levels[:, 1:]))) * (prices < levels[:, 1:]) * (prices_prev > levels[:, :-1]) * (~direction)
y_fee_density = (1./np.sqrt(np.maximum(prices, levels[:,:-1])) - 1./np.sqrt(np.minimum(prices_prev, levels[:, 1:])))*(prices<levels[:,1:])*(prices_prev > levels[:, :-1]) * (~direction)
y_fee_density_available = (1./np.sqrt(np.maximum(prices, levels[:,:-1])) - 1./np.sqrt(np.minimum(BIG_NUMBER, levels[:, 1:])))*(prices<levels[:,1:])#*(BIG_NUMBER > levels[:, 1:])

#y_fee_density_available = (np.sqrt(np.minimum(prices, levels[:,1:])) - np.sqrt(np.maximum(prices_prev, levels[:, :-1]))) * (prices > levels[:, :-1]) * (prices_prev < levels[:, 1:]) * direction

# COMPUTE THE TOTAL FEES IN BOTH TOKES STARTING FROM Nth MINT UNTIL THE PRICE MOVES OUTSIDE OF THE PRICE RANGE CORRESPONDING TO THAT MINT
# x_fees_collected and y_fees_collected are the total fees collected in the corresponding tokens
N = 3
mints = df['mint' == df.Type]
begin_range = mints['rangeLow'].iloc[N]
end_range = mints['rangeHigh'].iloc[N]
#liq_posted = np.min(histogram.loc[N,begin_range:end_range].head(-1))
price_in_range = (df.price < end_range) & (df.price > begin_range)
idx = mints.index[N]
1-(np.maximum(0., -price_in_range.astype('int').diff())).cummax()
price_in_range_terminal = price_in_range + 0
price_in_range_terminal.loc[:idx] = 1
price_in_range_terminal = price_in_range_terminal.cumprod()
is_swap = 'swap' == df['Type']
my_liquidity = mints['liquidity'].iloc[N]

x_amounts_available = pd.DataFrame(x_fee_density_available*histogram.iloc[:,:-1])*1e6
np.sum(x_amounts_available, axis=1)
y_amounts_available = pd.DataFrame(y_fee_density_available*histogram.iloc[:,:-1])*1e6
np.sum(y_amounts_available, axis=1)

x_fee_density_active = pd.DataFrame(x_fee_density[:,:], columns=dd3.iloc[:,:-1].columns).loc[:,begin_range:end_range] * (price_in_range*is_swap).values[:,np.newaxis]

my_histogram = pd.DataFrame(my_liquidity, index=histogram.index, columns=histogram.columns)

#x_fees_collected = np.sum(np.sum(x_fee_density_active*histogram, axis=0))*POOL_FEE*1e6
x_fees_collected = np.sum(np.sum(x_fee_density_active*my_histogram, axis=0))*POOL_FEE*1e6
x_fees_collected_t = np.sum(x_fee_density_active*my_histogram, axis=1)*POOL_FEE*1e6
x_fees_collected_running = np.cumsum(x_fees_collected_t)
x_fees_collected_total = np.sum(np.sum(x_fee_density_active*my_histogram, axis=0))*POOL_FEE*1e6
print('USDC fees collected after '+ str(N)+'th mint', x_fees_collected)

y_fee_density_active = pd.DataFrame(y_fee_density[:,:], columns=dd3.iloc[:,:-1].columns).loc[:,begin_range:end_range] * (price_in_range*is_swap).values[:,np.newaxis]

y_fees_collected = np.sum(np.sum(y_fee_density_active*my_histogram, axis=0))*POOL_FEE*1e6
y_fees_collected_t = np.sum(y_fee_density_active*my_histogram, axis=1)*POOL_FEE*1e6
y_fees_collected_running = np.cumsum(y_fees_collected_t)
print('ETH fees collected after '+ str(N)+'th mint', y_fees_collected)

my_total_usdc_investment = 1000.

p0 = df.price[idx]
my_liquidity = my_total_usdc_investment/(p0*np.sqrt(p0*end_range)/(end_range-p0) + np.sqrt(p0)-np.sqrt(begin_range))
my_histogram = pd.DataFrame(my_liquidity, index=histogram.index, columns=histogram.columns)
my_x_fees_collected = np.sum(np.sum(x_fee_density_active*my_histogram, axis=0))*POOL_FEE
a=0

# flag_x_tick = (df.pa != df.pa.shift(1)) | (df.pb != df.pb.shift(1))
# flag_x1_tick = flag_x_tick & ((df.pa == df.pb.shift(1)) | (df.pb == df.pa.shift(1)))
#
# np.sum(flag_x_tick)
# np.sum(flag_x1_tick)
#
# print("swaps crossing the tick", df[flag_x_tick & (df.Type == "swap")])
# print("swaps crossing one tick", df[flag_x1_tick & (df.Type == "swap")])
# print("swaps crossing one tick", df[flag_x_tick & (~flag_x1_tick) &(df.Type == "swap")])
