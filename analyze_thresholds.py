"""Analyze scenario label distribution under different thresholds."""
import sqlite3
import pandas as pd
import numpy as np

DB = r'C:\Users\HY\PycharmProjects\QuanTrade\QuanTrade\quant_system\data\quant.db'

conn = sqlite3.connect(DB)
df = pd.read_sql_query(
    'SELECT scenario_label_10d, target_return_10d, target_direction_10d FROM features_v5', conn
)
conn.close()

df['target_return_10d'] = pd.to_numeric(df['target_return_10d'], errors='coerce')
df['target_direction_10d'] = pd.to_numeric(df['target_direction_10d'], errors='coerce')
df = df.dropna(subset=['target_return_10d'])
r = df['target_return_10d']

N = len(r)

print('=' * 70)
print('CURRENT LABELS (adverse < -3%, favorable > +5%)')
print('=' * 70)
for lbl in ['adverse', 'base', 'favorable']:
    n = (df['scenario_label_10d'] == lbl).sum()
    print(f'  {lbl:10s}: {n:6d} ({n / N * 100:.1f}%)')

print()
print('RETURN STATS (10d forward)')
print(f'  N={N:,}  mean={r.mean() * 100:.2f}%  std={r.std() * 100:.2f}%  median={r.median() * 100:.2f}%')
print(f'  p10={r.quantile(0.1) * 100:.2f}%  p25={r.quantile(0.25) * 100:.2f}%  p75={r.quantile(0.75) * 100:.2f}%  p90={r.quantile(0.9) * 100:.2f}%')

print()
print('=' * 70)
print('BINARY LABEL (target_direction_10d)')
print('=' * 70)
d = df['target_direction_10d'].dropna()
print(f'  down(0): {(d == 0).sum():6d} ({(d == 0).sum() / len(d) * 100:.1f}%)')
print(f'  up  (1): {(d == 1).sum():6d} ({(d == 1).sum() / len(d) * 100:.1f}%)')

print()
print('=' * 70)
print('SCENARIO DISTRIBUTION UNDER DIFFERENT THRESHOLDS')
print('=' * 70)
base0 = ((r >= -0.03) & (r <= 0.05)).sum()
print(f'  {"fav_th":>8s} {"adv_th":>8s}  {"adv":>6s}({"%":>4s})   {"base":>6s}({"%":>4s})   {"fav":>6s}({"%":>4s})')
for fav_th, adv_th in [
    (0.05, -0.03), (0.04, -0.03), (0.03, -0.03), (0.03, -0.02),
    (0.02, -0.03), (0.02, -0.02), (0.01, -0.01)
]:
    fav = (r > fav_th).sum()
    adv = (r < adv_th).sum()
    base = N - fav - adv
    print(f'  >{fav_th * 100:+6.1f}% <{adv_th * 100:+6.1f}%  {adv:5d}({adv / N * 100:4.1f}%) {base:5d}({base / N * 100:4.1f}%) {fav:5d}({fav / N * 100:4.1f}%)')

print()
print('=' * 70)
print('WHAT DOES THE BINARY LABEL ACTUALLY MEAN?')
print('=' * 70)
print('  If target_direction_10d = (target_return_10d > 0):')
up_simple = (r > 0).sum()
dn_simple = (r <= 0).sum()
print(f'  up: {up_simple} ({up_simple / N * 100:.1f}%), down: {dn_simple} ({dn_simple / N * 100:.1f}%)')
print(f'  => Almost 50/50. No wonder binary acc ~50%.')
print()
print('  If we use target_direction_10d from DB directly:')
d = df['target_direction_10d'].dropna()
print(f'  up: {(d == 1).sum()}, down: {(d == 0).sum()}, ratio: {(d == 1).sum() / len(d) * 100:.1f}%')
