import numpy as np
import pandas as pd
import pyarrow.feather as feather

directory = r'C:\data\test'

TW0_POW_26 = 2**96

def abs_str(x):
    return x[1:] if '-' == x[0] else x

def prepare_data_swap(filename):
    df = pd.read_json(filename, lines=True)
    NN = len(df)
    out = pd.DataFrame(index=range(NN), columns=['blockNumber', 'logIndex', 'price', 'liquidity', 'USDCAmount', 'ETHAmount'])

    for ii in range(NN):
        if ii % 1000==0:
            print(ii) # monitor progress
        out.loc[ii, 'blockNumber'] = df.iloc[ii,:]['blockNumber']
        out.loc[ii, 'logIndex'] = df.iloc[ii,:]['logIndex']

        temp = int(df.iloc[ii,:]['args'][4], 16)/TW0_POW_26
        out.loc[ii, 'price'] = 1e12/temp**2
        out.loc[ii, 'liquidity'] = int(df.iloc[ii,:]['args'][5], 16)/1e18
        out.loc[ii, 'USDCAmount'] = int(abs_str(df.iloc[ii,:]['args'][2]), 16)/1e6
        out.loc[ii, 'ETHAmount'] = int(abs_str(df.iloc[ii,:]['args'][3]), 16)/1e18
    return out

def prepare_data_liquidity(filename, kk):
    df = pd.read_json(filename, lines=True)
    NN = len(df)
    out = pd.DataFrame(index=range(NN), columns=['blockNumber', 'logIndex', 'lowerTick', 'upperTick', 'liquidity', 'USDCAmount', 'ETHAmount']).astype({'lowerTick':'float','upperTick':'float'})
    for ii in range(NN):
        if ii % 1000 == 0:
            print(ii)  # monitor progress
        out.loc[ii, 'blockNumber'] = df.iloc[ii,:]['blockNumber']
        out.loc[ii, 'logIndex'] = df.iloc[ii,:]['logIndex']
        out.loc[ii, 'lowerTick'] = int(df.iloc[ii,:]['args'][2-kk])
        out.loc[ii, 'upperTick'] = int(df.iloc[ii,:]['args'][3-kk])
        out.loc[ii, 'liquidity'] = int(df.iloc[ii,:]['args'][4-kk], 16)/1e18
        out.loc[ii, 'USDCAmount'] = int(df.iloc[ii,:]['args'][5-kk], 16)/1e6
        out.loc[ii, 'ETHAmount'] = int(df.iloc[ii,:]['args'][6-kk], 16)/1e18
    out=out[out.liquidity>0]
    out['price'] = np.where(out['USDCAmount']*out['ETHAmount'] > 0, 1e12 / (np.sqrt(np.power(1.0001, out.lowerTick)) + out.ETHAmount / np.abs(out.liquidity)) ** 2, np.nan)
    out['rangeLow'] = 1e12 / np.power(1.0001, out.upperTick)
    out['rangeHigh'] = 1e12 / np.power(1.0001, out.lowerTick)
    return out

filename_mint = directory+'\\'+'mint.jsonl'
feather.write_feather(prepare_data_liquidity(filename_mint, 0), directory+'\\'+'mint')
print("processed", filename_mint)

filename_burn = directory+'\\'+'burn.jsonl'
feather.write_feather(prepare_data_liquidity(filename_burn, 1), directory+'\\'+'burn')
print("processed", filename_burn)

filename_swap = directory+'\\'+'swap_aa.jsonl'
feather.write_feather(prepare_data_swap(filename_swap), directory+'\\'+'swap_aa')
print("processed", filename_swap)

# BELOW SHOULD SIT IN A SEPARATE FILE

df_swap=feather.read_feather(directory+'\\'+'swap_aa')
df_mint = feather.read_feather(directory+'\\'+'mint')
df_burn = feather.read_feather(directory+'\\'+'burn')


df_swap['Type'] = 'swap'
df_mint['Type'] = 'mint'
df_burn['Type'] = 'burn'

out_df = pd.concat([df_swap, df_mint, df_burn], ignore_index=True).sort_values(['blockNumber', 'logIndex'])

