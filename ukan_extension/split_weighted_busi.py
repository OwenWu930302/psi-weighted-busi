"""
Keep only the weighted training images from weighted_cases_BUSI.csv.
Usage (in the repo folder): python ukan_extension/split_weighted_busi.py
Output: results/analysis/weighted_cases_BUSI_weighted.csv
"""
import pandas as pd

d = pd.read_csv('results/analysis/weighted_cases_BUSI.csv')
w = d[d.weighted == 'yes'].drop(columns='weighted')
w.to_csv('results/analysis/weighted_cases_BUSI_weighted.csv', index=False, encoding='utf-8-sig', float_format='%.4f')
print(w.groupby('expert').size().to_string())