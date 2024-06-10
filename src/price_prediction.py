import xgboost as xgb
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pyarrow.feather as feather
import gc
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler


BIG_NUMBER = 1e25
SMALL_NUMBER = 1/BIG_NUMBER
POOL_FEE = .0005

directory = "data"

# Code to "un-resample"
#df_merged = original_df.merge(df_upsampled, how='left', left_index=True, right_index=True)



# timestamps = feather.read_feather(directory+'\\'+'block_time_map').head(25000)
# readable_timestamps = np.array([datetime.utcfromtimestamp(int(ts)) for ts in timestamps.values if int(ts)!=0])
# timestamps = pd.Series(readable_timestamps, index=timestamps.index)

# Load in our data
df_mint = feather.read_feather(directory+'/'+'mint')
df_burn = feather.read_feather(directory+'/'+'burn')

df_swap = feather.read_feather(directory+'/'+'split_swaps/swap_ab')
# df_swap2 = feather.read_feather(directory+'/'+'split_swaps/swap_ab')



df_mint["Type"] = "Mint"
df_burn["Type"] = "Burn"
df_swap["Type"] = "Swap"
df_burn.loc[:, 'liquidity'] *= -1

num_swap_files = 3
# alph = "abcdefghijklmnopqrstuvwxyz"
alph = "bcdefghijklmnopqrstuvwxyz"

# Load multiple swap files
for i in range(1,1+num_swap_files):

    dir = directory+'/'+'split_swaps/swap_a' + alph[i]
    df_swap_add = feather.read_feather(dir)
    df_swap_add["Type"] = "Swap"
    df_swap = pd.concat([df_swap, df_swap_add], axis=0)

# Concatenate loaded data into one df
df_orig = pd.concat([df_swap, df_mint, df_burn], ignore_index=True).sort_values(['blockNumber', 'logIndex']).reset_index(drop=True)


# df = pd.concat([df_swap, df_swap2, df_mint, df_burn], ignore_index=True).sort_values(['blockNumber', 'logIndex']).reset_index(drop=True)
# flag_ts=df['blockNumber']<=timestamps.index[-1]
# df.loc[flag_ts,'timestamp'] = timestamps[df.loc[flag_ts,'blockNumber']].values

# df=df.head(100000).tail(70000).reset_index(drop=True)
# df=df.head(99000)

# print(df)
# df=df.tail(50000)
# df=df.drop(20000)
# df = df.iloc[20000:110000]
# df = df.head(100000)

# Calculate ranges in USD
df_orig['rangeLow'] = 1 / ( 1.0001**df_orig['upperTick'] / (10**12) )
df_orig['rangeHigh'] =  1 / ( 1.0001**df_orig['lowerTick'] / (10**12) )

# Filter by length of swap data
df = df_orig[(df_orig.timestamp <  df_swap["timestamp"].max()) & (df_orig.timestamp >  df_swap["timestamp"].min())]
# del df_mint, df_burn
del df_mint, df_burn, df_orig


gc.collect()
df['price'].ffill(inplace=True)
df['rangeLow'].ffill(inplace=True)
df['rangeHigh'].ffill(inplace=True)
df.loc[df.Type == 'swap', 'liquidity'] = 0


# Resample into 5 minute intervals
# df = df.resample('5T', on="timestamp").mean()

# remove any NAs
df_mask = df[["rangeLow", "rangeHigh"]].isna().any(axis=1)
df  = df[~df_mask]
df.reset_index(drop=True, inplace=True)


dd1= df.pivot_table(index=df.index, columns=['rangeLow'], values='liquidity', fill_value=0).cumsum() #Group together liquidity with same rangeLow, c
dd2 = df.pivot_table(index=df.index, columns=['rangeHigh'], values='liquidity', fill_value=0).cumsum()#Group together liquidity with same rangeHigh, c
dd3 = dd1.subtract(dd2, fill_value=0) # Subtract range highs and lows to only get net liquidity delta between each tick value
del dd1, dd2
gc.collect()

######Uncomment below
price_modelling = True
if price_modelling:
    histogram = dd3.cumsum(axis=1)

    # Resample on histogram directly instead of result_df

    # histogram["timestamp"] = df["timestamp"]
    # histogram["price"] = df["price"]

    # histogram = histogram.resample('0.5T', on="timestamp").mean().ffill()
    # price_resampled = histogram["price"] # maybe .copy
    # histogram.drop(columns=['price'], inplace=True)

    print(histogram)

    # Without resampling
    levels = dd3.columns.to_numpy()
    # prices = df['price'].to_numpy()
    prices = df['price'].groupby(df.index).mean().to_numpy()


    # With resampling on histogram
    # levels = dd3.columns.to_numpy()
    # prices = price_resampled.to_numpy()

    # Feature creation

    # Create a liquidity weighted price, for liquidity under price 20000
    col_idx = histogram.columns[histogram.columns < 20000]
    weighted_price = (histogram[col_idx]*col_idx).sum(axis=1) / histogram[col_idx].sum(axis=1)

    # Create a median price

    hist_probs = histogram[col_idx].div(histogram[col_idx].sum(axis=1), axis=0)

    # Function to calculate median value for each row
    def calculate_median(row):
        cumsum_probs = np.cumsum(row.values, axis=1)
        median_index = np.argmax(cumsum_probs >= 0.5,axis=1)
        return hist_probs.columns[median_index + 1]  # Add 1 to convert to 1-based indexing

    median_price = calculate_median(hist_probs).values

    # Standard deviation of liquidity density

    liquidity_std = np.sqrt(np.sum(hist_probs * (np.expand_dims(np.array(col_idx),axis=0) - np.expand_dims(np.array(weighted_price),axis=1))**2, axis=1))
    liquidity_std.ffill(inplace=True)


    indices = np.digitize(prices, levels, right=True)

    # Create local liquidity weighted prices

    depths = [3,10,50] # Number of ticks included in weighting on either side of current price

    for d in depths:
        var_name = "local_weighted_price_" + str(d)
        globals()[var_name] = [(histogram.iloc[i, col-d:col+d]*levels[col-d:col+d]).sum() / histogram.iloc[i, col-d:col+d].sum() for i, col in enumerate(indices)]
        globals()[var_name] = np.array(globals()[var_name])

    # Create dataframe of features for modelling
    result_df = pd.concat([(df.price-local_weighted_price_3)/df.price,
                            (df.price-local_weighted_price_10)/df.price, 
                            (df.price-local_weighted_price_50)/df.price, 
                            (df.price-weighted_price)/df.price, 
                            (df.price-median_price)/df.price,
                            liquidity_std], axis=1)

    result_df.columns = ['WeightedPrice_a', 
                        'WeightedPrice_b', 
                        'WeightedPrice_c',
                        'WeightedPrice_all',
                        "MedianPrice", 
                        'LiquidityStd']

    result_df["price"] = df["price"]
    result_df.index = df.timestamp

    # We carry out resampling to get our desired time intervals
    freq = "1min"
    res_resampled = result_df.resample(freq).mean().dropna()

    res_resampled.ffill(inplace=True)

    res_resampled["logPriceRet"] = np.log(res_resampled["price"]).diff()

    res_resampled = res_resampled.dropna()

    # Trim the data due to the shift in volatility regime
    res_resampled = res_resampled[res_resampled.index > "2021-06-02"]


    lookback = 4 # Number of historical timesteps we use to predict the future price

    resampled_price = res_resampled["price"]
    input_df = res_resampled.drop(columns=['price'])

    # Uncomment below for price pred to work

    columns_to_drop = ['WeightedPrice_a', 'WeightedPrice_b', 'WeightedPrice_c']
    input_df = input_df.drop(columns=columns_to_drop)

    # We include the data from past timesteps
    shifted_df = pd.concat([input_df.shift(i) for i in range(0, lookback + 1)], axis=1)
    # shifted_df = pd.concat([result_df.shift(i) for i in range(0, lookback + 1)], axis=1)

    # Rename the columns with the appropriate suffixes
    new_cols = []
    for i in range(lookback+1):
        for col in input_df.columns:
            new_cols.append(str(col)+str(i))

    shifted_df.columns = new_cols


    lookahead = 1
    y_ret = res_resampled["logPriceRet"].shift(-lookahead)

    x_y_df = pd.concat([pd.DataFrame(y_ret),shifted_df], axis = 1).dropna()

    # Create our feature data frame and target data frame
    X = x_y_df.iloc[:,1:]
    y = x_y_df.iloc[:,0]

    # Split data into train and test
    test_size = 0.2
    split_index = int(X.shape[0] * (1 - test_size))

    X_train, X_test, y_train, y_test = X.iloc[:split_index], X.iloc[split_index:], y.iloc[:split_index], y.iloc[split_index:]
    resampled_price_test = resampled_price.iloc[split_index: ]

    scaler = StandardScaler()
    # We scale feature data
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = xgb.XGBRegressor(
        n_estimators=2000, 
        learning_rate=0.01,
        min_child_weight = 1,
        max_depth=1,
        colsample_bytree=0.7,
        subsample=0.7,
        reg_alpha=0.1,
        reg_lambda=0.01
    )

    # Train the model
    model.fit(X_train_scaled, y_train)
    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Test results
    mse = mean_squared_error(y_test, y_pred)
    model_r2 = r2_score(y_test, y_pred)
    raw_r2  = np.corrcoef(y_test.values, y_pred)[0,1]**2

    print("Test Mean Squared Error:", mse)
    print("Test Model R2 Score: ", model_r2)
    print("Test Raw R2 Score: ", raw_r2)


    accuracy = np.sum(np.array( np.sign(y_test.values) == np.sign(y_pred)) )/ y_test.shape[0]
    print("Accuracy:", accuracy)
    print("Best Static Guess: ", max(np.sum(np.sign(y_test.values) == 1) / y_test.shape[0], 1 - np.sum(np.sign(y_test.values) == 1) / y_test.shape[0]))
    print("y_test", y_test.shape, y_test)

###########Uncomment above


# Volatility Modelling
vol=True
if vol:

    lookback = 1 # Number of historical timesteps we use to predict the future price
    lookahead = 60


    vol_input_df = res_resampled.drop(columns=['price'])
    vol_input_df["logPriceRet_sq"] = np.sqrt((vol_input_df["logPriceRet"]**2))

    vol_input_df["rolling_vol"] = np.sqrt((res_resampled["logPriceRet"]**2).rolling(lookahead).sum()) # Vol over past lookahead intervals

    # columns_to_drop = ['logPriceRet', 'WeightedPrice_a', 'WeightedPrice_b', 'WeightedPrice_c']
    # columns_to_drop = ['WeightedPrice_a', 'WeightedPrice_b', 'WeightedPrice_c']
    columns_to_drop = []
    vol_input_df = vol_input_df.drop(columns=columns_to_drop)

    vol_shifted_df = pd.concat([vol_input_df.shift(i) for i in range(0, lookback + 1)], axis=1)
    # shifted_df = pd.concat([result_df.shift(i) for i in range(0, lookback + 1)], axis=1)

    # Rename the columns with the appropriate suffixes
    new_cols = []
    for i in range(lookback+1):
        for col in vol_input_df.columns:
            new_cols.append(str(col)+str(i))

    vol_shifted_df.columns = new_cols

    y_vol = np.sqrt((res_resampled["logPriceRet"]**2).rolling(lookahead).sum()).shift(-lookahead)*10e2 # We predict total vol over next lookahead intervals as it is less noisy

    x_y_vol_df = pd.concat([pd.DataFrame(y_vol),vol_shifted_df], axis = 1).dropna()

    X_vol = x_y_vol_df.iloc[:,1:]
    y_vol = x_y_vol_df.iloc[:,0]

    test_size = 0.2
    split_index = int(X_vol.shape[0] * (1 - test_size))

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=20)

    X_vol_train, X_vol_test, y_vol_train, y_vol_test = X_vol.iloc[:split_index], X_vol.iloc[split_index:], y_vol.iloc[:split_index], y_vol.iloc[split_index:]

    vol_scaler = StandardScaler()

    X_vol_train_scaled = vol_scaler.fit_transform(X_vol_train)
    X_vol_test_scaled = vol_scaler.transform(X_vol_test)

    vol_model = xgb.XGBRegressor(
        n_estimators=500, 
        learning_rate=0.01,
        min_child_weight = 5,
        max_depth=2,
        colsample_bytree=0.7,
        subsample=0.7,
        reg_alpha=0.5*10e3,
        reg_lambda=0.5*10e3
    )

    vol_model.fit(X_vol_train_scaled, y_vol_train, 
        verbose=False
    )

    print("X_vol_test_scaled and y_vol_test", X_vol_test_scaled.shape, y_vol_test.shape, X_vol_test_scaled, y_vol_test)

    # Make predictions
    y_vol_pred = vol_model.predict(X_vol_test_scaled)

    mse = mean_squared_error(y_vol_test, y_vol_pred)
    model_r2 = r2_score(y_vol_test, y_vol_pred)
    raw_r2  = np.corrcoef(y_vol_test.values, y_vol_pred)[0,1]**2
    
    # print("y_vol_test", y_vol_test.shape, y_vol_test)

    print("Test Mean Squared Error:", mse)
    print("Test Model R2 Score: ", model_r2)
    print("Test Raw R2 Score: ", raw_r2)

# We need to cut first 63 rows of price predictions and last 3 rows so it lines up with vol predictions


# Poisson Process modelling

# We assume our frequency is a "unit" of time

from sklearn import linear_model
# clf = linear_model.PoissonRegressor()
# clf = xgb.XGBRegressor()

poisson=False
if poisson: # Poisson regression to learn relationship between volatilty and number of swaps in an interval

    df_mint = feather.read_feather(directory+'/mint')
    df_burn = feather.read_feather(directory+'/burn')

    df_swap = feather.read_feather(directory+'/split_swaps/swap_ab')

    df_mint["Type"] = "Mint"
    df_burn["Type"] = "Burn"
    df_swap["Type"] = "Swap"
    df_burn.loc[:, 'liquidity'] *= -1

    num_swap_files = 3
    alph = "bcdefghijklmnopqrstuvwxyz"

    for i in range(1,1+num_swap_files):

        dir = directory+'/split_swaps/swap_a' + alph[i]
        df_swap_add = feather.read_feather(dir)
        df_swap_add["Type"] = "Swap"
        df_swap = pd.concat([df_swap, df_swap_add], axis=0)

    freq = "5min"

    result_df["logretSq"] = result_df["price"].pct_change()**2

    vol_resampled = result_df["logretSq"].resample(freq).sum().dropna()

    vol_resampled.ffill(inplace=True)

    df_swap = df_swap.set_index(df_swap.timestamp)
    swap_counts = df_swap["Type"].resample(freq).count()
    liquidity_sums = df_swap["liquidity"].resample(freq).sum()

    vol_and_swaps = pd.concat([vol_resampled, swap_counts, liquidity_sums], axis=1)
    vol_and_swaps = vol_and_swaps[vol_and_swaps.index > "2021-06-02"].dropna()
    x = np.sqrt(vol_and_swaps["logretSq"])
    y = vol_and_swaps["Type"]

def simulate_poisson_jump_times(rate, T):
    """
    Simulate jump times according to a Poisson process with rate lambda over time period T.
    """
    jump_times = [0] # Start with 0 as so that simulate gbm function works
    current_time = 0
    
    while current_time < T:
        # Generate the next inter-arrival time
        inter_arrival_time = np.random.exponential(1 / rate)
        current_time += inter_arrival_time
        
        if current_time < T:
            jump_times.append(current_time)
    
    return np.array(jump_times)

def simulate_poisson_jump_times_vec(rate, T):
    """
    Simulate jump times according to a Poisson process with rate lambda over time period T.
    Vectorized version.
    """
    # Estimate the number of potential jumps
    num_jumps = int(rate * T * 1.5)
    
    # Generate inter-arrival times
    inter_arrival_times = np.random.exponential(1 / rate, num_jumps)
    
    # Compute jump times
    jump_times = np.cumsum(inter_arrival_times)
    
    # Filter out the times beyond T
    jump_times = jump_times[jump_times < T]
    
    return np.insert(jump_times, 0, 0)

def simulate_gbm_at_jump_times(S0, mu, sigma, jump_times):
    """
    Simulate the GBM at the specified jump times.
    """
    # Initialize the price list with the initial price
    prices = [S0]
    
    for i in range(1, len(jump_times)):
        # Calculate the time difference
        delta_t = jump_times[i] - jump_times[i - 1]
        
        # Generate a standard normal random variable
        Z = np.random.normal(0, 1)
        
        # Update the price using the GBM solution
        S_prev = prices[-1]
        S_new = S_prev * np.exp((mu - 0.5 * sigma**2) * delta_t + sigma * np.sqrt(delta_t) * Z)
        prices.append(S_new)
    
    return prices

def simulate_gbm_with_predictions(S0, predicted_log_returns, predicted_volatilities, jump_times):
    """
    Simulate the GBM at the specified jump times using predicted log returns and volatilities.
    Vectorized version.
    """
    # Calculate the time differences (delta_t)
    delta_ts = np.diff(jump_times, prepend=0)
    
    # Generate standard normal random variables for each time step
    Z = np.random.normal(0, 1, len(delta_ts))
    
    # Calculate the exponent terms in the GBM formula
    exponent_terms = (predicted_log_returns - 0.5 * predicted_volatilities**2) * delta_ts + predicted_volatilities * np.sqrt(delta_ts) * Z
    
    # Compute the cumulative sum of exponents
    cum_exponents = np.cumsum(exponent_terms)
    
    # Calculate the prices
    prices = S0 * np.exp(cum_exponents)
    
    return prices


# Attempts with real data
# pred_vols = np.array(y_vol_pred)
# pred_logrets = np.array(y_pred)
# pred_rates = np.array(yp_pred)
# current_prices = np.array(resampled_price_test)
# print("Shapes: ", pred_vols.shape, pred_logrets.shape, pred_rates.shape, current_prices.shape)

# jump_times_list = [] # To contain the different simulations
# price_paths_list = []

# jump_times = np.apply_along_axis(simulate_poisson_jump_times_vec, axis=0, arr=pred_rates) # Contains a simulated jump times for next interval for each interval
# price_paths = simulate_gbm_with_predictions(current_prices, pred_logrets, pred_rates, jump_times) # Containts a simulated price path for next interval for each interval

# plt.figure(figsize=(10, 6))
# plt.plot(jump_times[-1], price_paths[-1], marker='o')
# plt.xlabel("Time")
# plt.ylabel("Price")
# plt.title("Geometric Brownian Motion with Poisson Jumps")
# plt.legend()
# plt.grid(True)
# plt.show()





plot_sample_paths=False
if plot_sample_paths:
    n_sims = 10 # Number of simulations
    jump_times_list=[]
    paths_list = []
    for i in range(0,n_sims):
        jump_times = simulate_poisson_jump_times(5,1)

        # price_path = simulate_gbm_at_jump_times(100, .01, 0.05, jump_times)
        price_path = simulate_gbm_with_predictions(3000, .01, 0.05, jump_times)

        jump_times_list.append(jump_times)
        paths_list.append(price_path)


    # Plot the results
    plt.figure(figsize=(10, 6))
    for i in range(0,n_sims):   
        plt.step(jump_times_list[i], paths_list[i], where="post", marker='o')
    plt.xlabel("Time")
    plt.ylabel("Price")
    plt.title("Geometric Brownian Motion with Poisson Jumps")
    plt.legend()
    plt.grid(True)
    plt.show()










