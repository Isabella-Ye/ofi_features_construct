import pandas as pd
import numpy as np
from sklearn.decomposition import PCA


df = pd.read_csv('/Users/isabella/Downloads/first_25000_rows.csv')
df = df.sort_values('ts_event').reset_index(drop=True)
print(df.head())

# 1. compute best-level ofi (level 0)
bid_px0 = df['bid_px_00'].values
ask_px0 = df['ask_px_00'].values
bid_sz0 = df['bid_sz_00'].values
ask_sz0 = df['ask_sz_00'].values

prev_bid_px0 = np.empty_like(bid_px0)
prev_ask_px0 = np.empty_like(ask_px0)
prev_bid_sz0 = np.empty_like(bid_sz0)
prev_ask_sz0 = np.empty_like(ask_sz0)
prev_bid_px0[0], prev_ask_px0[0] = bid_px0[0], ask_px0[0]
prev_bid_sz0[0], prev_ask_sz0[0] = bid_sz0[0], ask_sz0[0]
prev_bid_px0[1:], prev_ask_px0[1:] = bid_px0[:-1], ask_px0[:-1]
prev_bid_sz0[1:], prev_ask_sz0[1:] = bid_sz0[:-1], ask_sz0[:-1]

ofi_bid0 = np.where(bid_px0 > prev_bid_px0, bid_sz0,
            np.where(bid_px0 < prev_bid_px0, -bid_sz0,
                     bid_sz0 - prev_bid_sz0))
ofi_ask0 = np.where(ask_px0 > prev_ask_px0, ask_sz0,
            np.where(ask_px0 < prev_ask_px0, -ask_sz0,
                     ask_sz0 - prev_ask_sz0))
best_ofi = ofi_bid0 - ofi_ask0

# 2. compute Multi-Level ofi (sum for levels 0-2)
levels = [0, 1, 2]
ofi_bid_levels = np.zeros((len(df), len(levels)))
ofi_ask_levels = np.zeros((len(df), len(levels)))
for i, lvl in enumerate(levels):
    # get current and previous price/size for each level
    bid_px = df[f'bid_px_0{lvl}'].values
    ask_px = df[f'ask_px_0{lvl}'].values
    bid_sz = df[f'bid_sz_0{lvl}'].values
    ask_sz = df[f'ask_sz_0{lvl}'].values
    prev_bid_px = np.empty_like(bid_px)
    prev_ask_px = np.empty_like(ask_px)
    prev_bid_sz = np.empty_like(bid_sz)
    prev_ask_sz = np.empty_like(ask_sz)
 
    prev_bid_px[0], prev_ask_px[0] = bid_px[0], ask_px[0]
    prev_bid_sz[0], prev_ask_sz[0] = bid_sz[0], ask_sz[0]
    prev_bid_px[1:], prev_ask_px[1:] = bid_px[:-1], ask_px[:-1]
    prev_bid_sz[1:], prev_ask_sz[1:] = bid_sz[:-1], ask_sz[:-1]

    # compute ofi for this level
    ofi_bid = np.where(bid_px > prev_bid_px, bid_sz,
               np.where(bid_px < prev_bid_px, -bid_sz,
                        np.nan_to_num(bid_sz - prev_bid_sz, nan=0.0)))
    ofi_ask = np.where(ask_px > prev_ask_px, ask_sz,
               np.where(ask_px < prev_ask_px, -ask_sz,
                        np.nan_to_num(ask_sz - prev_ask_sz, nan=0.0)))
    # store results
    ofi_bid_levels[:, i] = np.nan_to_num(ofi_bid, nan=0.0)
    ofi_ask_levels[:, i] = np.nan_to_num(ofi_ask, nan=0.0)

# then sum bid-minus-ask OFI across levels 0-2
multi_ofi = ofi_bid_levels.sum(axis=1) - ofi_ask_levels.sum(axis=1)

# 4. compute Integrated OFI via PCA on multi-level OFIs

# Construct multi-level OFI difference vector for each timestamp
ofi_diff_levels = ofi_bid_levels - ofi_ask_levels  # with shape (N, 3)

# Normalize by average depth at each level per timestamp
norm_ofi = np.zeros_like(ofi_diff_levels)
for i, lvl in enumerate(levels):
    depth_avg = (df[f'bid_sz_0{lvl}'].values + df[f'ask_sz_0{lvl}'].values) / 2.0
    # avoid division by zero (if depth_avg is 0, set normalized OFI to 0)
    norm_ofi[:, i] = np.where(depth_avg != 0, ofi_diff_levels[:, i] / depth_avg, 0.0)

# then we apply PCA on normalized multi-level OFI vectors
pca = PCA(n_components=1)
pca.fit(norm_ofi)
first_pc = pca.components_[0]  

# Normalize PC weights by L1 norm (sum of absolute values)
w1 = first_pc
w1_l1 = w1 / np.sum(np.abs(w1))

# Compute integrated OFI as projection onto first PC (with L1-normalized weights)
integrated_ofi = norm_ofi.dot(w1_l1)

# 5. since the dataset only includes one symbol - AAPL, I decide to simulate SPY (as a benchmark) to compute cross-asset ofi. 
#  i want to measure how ofi in AAPL deviates from that of SPY
N = len(df)
# compute AAPL mid-price changes to estimate volatility
aapl_mid = (df['bid_px_00'].values + df['ask_px_00'].values) / 2.0
aapl_mid_diff = np.diff(aapl_mid, prepend=aapl_mid[0])
vol = np.nanstd(aapl_mid_diff)  # standard deviation of mid-price changes
np.random.seed(42)  # for reproducibility

# then, simulate SPY mid-price as random walk with matching volatility
spy_mid = np.zeros(N)
spy_mid[0] = 400.0  # starting price for SPY (arbitrary)

# random price changes ~ Normal(0, vol)
spy_diffs = np.random.normal(loc=0.0, scale=vol, size=N-1)
spy_mid[1:] = spy_mid[0] + np.cumsum(spy_diffs)

# define SPY best bid/ask from mid (using a fixed small spread, here i use 0.02)
spread = 0.02
spy_bid_px = spy_mid - spread/2
spy_ask_px = spy_mid + spread/2
# simulate SPY best sizes by sampling from AAPL's level-0 size distribution
spy_bid_sz = np.random.choice(bid_sz0, size=N, replace=True)
spy_ask_sz = np.random.choice(ask_sz0, size=N, replace=True)
# compute SPY best-level OFI using same logic as AAPL
prev_spy_bid_px = np.empty_like(spy_bid_px)
prev_spy_ask_px = np.empty_like(spy_ask_px)
prev_spy_bid_sz = np.empty_like(spy_bid_sz)
prev_spy_ask_sz = np.empty_like(spy_ask_sz)
prev_spy_bid_px[0], prev_spy_ask_px[0] = spy_bid_px[0], spy_ask_px[0]
prev_spy_bid_sz[0], prev_spy_ask_sz[0] = spy_bid_sz[0], spy_ask_sz[0]
prev_spy_bid_px[1:], prev_spy_ask_px[1:] = spy_bid_px[:-1], spy_ask_px[:-1]
prev_spy_bid_sz[1:], prev_spy_ask_sz[1:] = spy_bid_sz[:-1], spy_ask_sz[:-1]
ofi_spy_bid = np.where(spy_bid_px > prev_spy_bid_px, spy_bid_sz,
               np.where(spy_bid_px < prev_spy_bid_px, -spy_bid_sz,
                        spy_bid_sz - prev_spy_bid_sz))
ofi_spy_ask = np.where(spy_ask_px > prev_spy_ask_px, spy_ask_sz,
               np.where(spy_ask_px < prev_spy_ask_px, -spy_ask_sz,
                        spy_ask_sz - prev_spy_ask_sz))
spy_best_ofi = ofi_spy_bid - ofi_spy_ask

# compute Cross-Asset OFI as AAPL best-level OFI minus SPY best-level OFI
cross_asset_ofi = best_ofi - spy_best_ofi

# 6. Save results to a new CSV file
output = pd.DataFrame({
    'ts_event': df['ts_event'],
    'best_ofi': best_ofi,
    'multi_ofi': multi_ofi,
    'integrated_ofi': integrated_ofi,
    'cross_asset_ofi': cross_asset_ofi
})
output.to_csv('ofi_features.csv', index=False)