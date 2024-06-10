import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pyarrow.feather as feather

#"what would be the yield had I staked $X (total dollar amount of both tokens) equivalent  on day d1 (so there will be a price p1) at the range (p_a, p_b) and remove it on d2"
# Uniswap V3 USDC/ETH 5bps pool


p_a = 4998.918171 # Should be higher than p_b
p_b = 2499
dollars=6.438842e+03 # Amount of dollars staked

# Get tick range
# i_a = np.floor( np.emath.logn(1.0001, (1/p_a)*10**12) )# Lower   #Tick must be multiple of 10 for this pool
# i_b = np.floor( np.emath.logn(1.0001, (1/p_b)*10**12) )# Upper

i_a = 10*round( np.emath.logn(1.0001, (1/p_a)*10**12) /10 )# Lower   #Tick must be multiple of 10 for this pool
i_b = 10*round( np.emath.logn(1.0001, (1/p_b)*10**12) /10 )# Upper

ourPositionTicks = np.arange(i_a, i_b + 1, 10)
#print(ourPositionTicks)

# Convert tick back to exact prices to use as keys with dataframes
p_a = 1 / ( 1.0001**i_a / (10**12) )
p_b = 1 / ( 1.0001**i_b / (10**12) )

#print(p_a_real,p_b_real)

"""For technical reasons explained in 6.2.1, however, pools actually track
ticks at every square root price"""

directory = "data"
df_mint = feather.read_feather(directory+'/'+'mint')
df_burn = feather.read_feather(directory+'/'+'burn')

df_swap = feather.read_feather(directory+'/'+'split_swaps/swap_aa')


# We calculate our initial liquidity, this only works for p_a < price < p_b at the moment
initial_price_usd = 3443.333910 # hardcode from mints/burns data for now
initial_price = 1/initial_price_usd
L_x = dollars * ( np.sqrt(initial_price*(1/p_b)) ) / ( np.sqrt((1/p_b)) - np.sqrt(initial_price   ))
y = L_x*(np.sqrt(initial_price) - np.sqrt(1/p_a))
dollar_ratio = dollars / (dollars + y*initial_price_usd)

dollars_provided = dollar_ratio*dollars
eth_provided = (1-dollar_ratio)*dollars/initial_price_usd
L_y = eth_provided / (np.sqrt(initial_price) - np.sqrt(1/p_a))
L = np.min([L_x*dollar_ratio, L_y])/10**6 # Our liquidity

print("Amount of USDC deposited in pool is: ", dollars_provided)
print("Amount of ETH deposited in pool is: ", eth_provided)
print("Liquidity is:",L)


df_mint['Type'] = 'mint'
df_burn['Type'] = 'burn'
df_burn['liquidity'] = -df_burn['liquidity'] 

see = pd.concat([df_mint, df_burn], ignore_index=True).sort_values(['blockNumber', 'logIndex'])
see= see.head(250)
see['rangeLow'] = 1 / ( 1.0001**see['upperTick'] / (10**12) )
see['rangeHigh'] =  1 / ( 1.0001**see['lowerTick'] / (10**12) )
see['positionValueUSD']  = see['USDCAmount'] + see['ETHAmount']*see['price']

# filtered_rows = df[(df['B'] > condition_lower) & (df['C'] < condition_upper)]
inRangeTotalLiquidity = see[ (see["rangeHigh"]> p_b ) | (see["rangeLow"]> p_a )  ]["liquidity"].cumsum()
see["ourLiquidityShare"] = L / (inRangeTotalLiquidity+L)
# df.loc[filtered_rows.index, 'Result'] = filtered_rows['A']
# see.loc[inRangeTotalLiquidity.index, "ourLiquidityShare"] = L / ( inRangeTotalLiquidity + L)

# print(see[30:50])
#print(df_swap)

# BIG_NUMBER = 1e25
# #print(see)

# price = see['price'].to_numpy()

# dd1= see.pivot_table(index=see.index, columns=['rangeLow'], values='liquidity', fill_value=0).cumsum() #Group together liquidity with same rangeLow, c 
# ind1 = (0 == dd1).cumprod().astype('int')


# dd2 = see.pivot_table(index=see.index, columns=['rangeHigh'], values='liquidity', fill_value=0).cumsum()
# ind2 = (0 == dd2).cumprod().astype('int')

# dd3=dd1.subtract(dd2, fill_value=0) # Subtract range highs and lows to only get net liquidity delta between each tick value
# ind3 = (0 == dd3).cumprod().astype('int')
# histogram_out = dd3.cumsum(axis=1) # Sum liquidity deltas to get actual amount of liquidity between each value


# # determine the intialized tick range where the current price is
# B = np.where((price[:, np.newaxis])*(-BIG_NUMBER*ind3+1-ind3) > dd3.columns.to_numpy()[np.newaxis,:], 1, 0)
# price_lt = -((price[:, np.newaxis] - dd3.columns.to_numpy()[np.newaxis,:])*B + BIG_NUMBER*(1-B)).min(axis=1)+price

# B1 = np.where((price[:, np.newaxis])*(BIG_NUMBER*ind3+1-ind3) <= dd3.columns.to_numpy()[np.newaxis,:], 1, 0)
# price_rt = ((dd3.columns.to_numpy()[np.newaxis,:]-price[:, np.newaxis])*B1+BIG_NUMBER*(1-B1)).min(axis=1)+price

# def find_closest_values(arr, l, u):
#     sorted_indices = np.argsort(arr)
#     sorted_arr = arr[sorted_indices]

#     closest_below_index = np.searchsorted(sorted_arr, l, side='right') - 1
#     closest_below = sorted_arr[closest_below_index] if closest_below_index >= 0 else None

#     closest_above_index = np.searchsorted(sorted_arr, u, side='left')
#     closest_above = sorted_arr[closest_above_index] if closest_above_index < len(arr) else None

#     return closest_below, closest_above

# l_ticks = dd3.columns.to_numpy()
# #print(l_ticks)

# l, u = find_closest_values(l_ticks, p_a, p_b)

# liquidity_between_ticks = dd3.loc[:, u:l]
# liquidity_between_ticks = liquidity_between_ticks.sum(axis=1) # This is a dataframe showing total liquidity in the range of our position
#print(see)

#print(liquidity_between_ticks)

#print(dd3)
# #print(dd3["saafsf"])
# print(see)
# print(df_swap)