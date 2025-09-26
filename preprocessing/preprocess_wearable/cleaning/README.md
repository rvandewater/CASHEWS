# Data Cleaning Process

1. Remove duplicate timestamps rows (keep first)
2. Segment data at longer gaps (`--time_gaps_s` (default > 60s))
3. For each segment do
    1. Apply signal filter (default 4th-order Chebyshev type II filter with bandpass cutoff at \[0.5, 5\])
    2. Potentially remove and interpolate areas with high acceleration (motion artifacts). If activated, removes areas where acceleration magnitude > acceleration mean + std. Then interpolates using monotone cubic interpolation (neurokit2)
    3. Expand time series to complete datetime with regular time intervals between rows, set added rows to NaN
4. Reassemble dataframe and save to directory
