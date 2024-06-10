import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import pyarrow.feather as feather

# Price is quantity of token1 / quantity of token0
# Liq is divided by 10^18

# Uniswap V3 USDC/ETH 5bps pool
# sqrtpriceX96 is sqrt(price) * 2**96    (in decimal representation)

#Histogram oesn't account for swaps changing liquidity

#mints = pd.read_json('mint.jsonl', lines=True)
#see = pd.read_csv("data/good_pivot_full.csv", index_col=0) #First ever transactions on this uniswapv3 pool


directory = "data"
df_mint = feather.read_feather(directory+'/'+'mint')
df_burn = feather.read_feather(directory+'/'+'burn')

df_swap = feather.read_feather(directory+'/'+'split_swaps/swap_aa')


df_mint['Type'] = 'mint'
df_burn['Type'] = 'burn'
df_burn['liquidity'] = -df_burn['liquidity'] 

see = pd.concat([df_mint, df_burn], ignore_index=True).sort_values(['blockNumber', 'logIndex'])
see= see.head(250)
see['rangeLow'] = 1 / ( 1.0001**see['upperTick'] / (10**12) )
see['rangeHigh'] =  1 / ( 1.0001**see['lowerTick'] / (10**12) )

print(see)
print(df_swap)

BIG_NUMBER = 1e25

price = see['price'].to_numpy()

dd1= see.pivot_table(index=see.index, columns=['rangeLow'], values='liquidity', fill_value=0).cumsum() #Group together liquidity with same rangeLow, c 
ind1 = (0 == dd1).cumprod().astype('int')

print(dd1)
dd2 = see.pivot_table(index=see.index, columns=['rangeHigh'], values='liquidity', fill_value=0).cumsum()
ind2 = (0 == dd2).cumprod().astype('int')

dd3=dd1.subtract(dd2, fill_value=0) # Subtract range highs and lows to only get net liquidity delta between each tick value
ind3 = (0 == dd3).cumprod().astype('int')
histogram_out = dd3.cumsum(axis=1) # Sum liquidity deltas to get actual amount of liquidity between each value


# determine the intialized tick range where the current price is
B = np.where((price[:, np.newaxis])*(-BIG_NUMBER*ind3+1-ind3) > dd3.columns.to_numpy()[np.newaxis,:], 1, 0)
price_lt = -((price[:, np.newaxis] - dd3.columns.to_numpy()[np.newaxis,:])*B + BIG_NUMBER*(1-B)).min(axis=1)+price

B1 = np.where((price[:, np.newaxis])*(BIG_NUMBER*ind3+1-ind3) <= dd3.columns.to_numpy()[np.newaxis,:], 1, 0)
price_rt = ((dd3.columns.to_numpy()[np.newaxis,:]-price[:, np.newaxis])*B1+BIG_NUMBER*(1-B1)).min(axis=1)+price


histograms = histogram_out.to_numpy()
num_histograms = histograms.shape[0]

# Amounts to taper off of left and right sides of histogram to basically "zoom in"
l_taper  = 5
r_taper = 10

# Some liquidity values are < zero
print(dd3)
histograms[histograms < 0] = 0 

l_ticks = dd3.columns.to_numpy()

fig, ax = plt.subplots(1,1,figsize=(20, 5))

ax.plot(l_ticks[l_taper:-r_taper], histograms[0][l_taper:-r_taper])
ax.grid()
ax.set_xlabel("Price")
ax.set_ylabel("Liquidity")
#ax.axvline(price[0],color="red", ymin=0)


def update(frame):
    
    ax.clear()
    ax.plot(l_ticks[l_taper:-r_taper], histograms[frame][l_taper:-r_taper])
    ax.grid()
    ax.set_title(f'Time Step: {frame + 1}')
    ax.fill_between(l_ticks[0:-10], histograms[frame][0:-10])
    #ax[0].axvline( price_lt[frame], color="red", ymin=0, label="Price Lower Tick: "+str(price_lt[frame]))
    ax.stem(price_lt[frame], histograms[frame][l_taper:-r_taper].max(), linefmt="r--", markerfmt=" ", label="Price Lower Tick: "+str(price_lt[frame]))
    ax.set_xlabel("Price")
    ax.set_ylabel("Liquidity")
    ax.legend(loc="upper right")

showPlot=False

if showPlot:
    # Set up the animation
    animation = FuncAnimation(fig, update, frames=num_histograms, interval=10, repeat=False)
    plt.show()

#print(price_lt)
print(histograms.shape)
#print()
#dd4.to_clipboard()

