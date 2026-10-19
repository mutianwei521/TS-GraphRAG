"""Frozen allocation observations, with complete-draw summary checks."""
import numpy as np
import pandas as pd
from nature_style import ROOT
E5 = ROOT / 'results' / 'E5_design_curve_20260909'
SUMMARY = pd.read_csv(E5/'e5_h0_summary.csv')
DRAWS = pd.read_csv(E5/'e5_h0_draws.csv')
PAIRS = pd.read_csv(E5/'e5_g2_top1_pairwise.csv')
DES = ['A', 'A_prime', 'C60', 'C90', 'B']
LABELS = ['A\n30 × 6', 'A′\n30 × 6', 'C60\n60 × 3', 'C90\n90 × 2', 'B\n180 × 1']
COL = {'top1':'top1_percent', 'top3':'top3_percent', 'mrr':'mrr_at5_percent', 'med_m':'median_error_m'}
def values(group, design, metric):
    rows = DRAWS[(DRAWS.group==group)&(DRAWS.design==design)&(DRAWS.protocol=='H0')].sort_values('actual_seed')
    vals = rows[COL[metric]].to_numpy(float)
    row = SUMMARY[(SUMMARY.group==group)&(SUMMARY.design==design)&(SUMMARY.protocol=='H0')&(SUMMARY.metric==metric)]
    assert len(row)==1 and len(vals)==15 and rows.actual_seed.nunique()==15
    assert np.isfinite(vals).all()
    assert abs(vals.mean()-float(row.iloc[0]['mean'])) < .06
    assert abs(vals.std(ddof=1)-float(row.iloc[0]['sample_sd'])) < .06
    return vals
def stat(group, design, metric):
    v=values(group,design,metric)
    return v.mean(),v.std(ddof=1),v
def pair(first,second):
    pcol=next(c for c in PAIRS.columns if c.startswith('p_holm'))
    mask=(PAIRS.group=='G2')&(PAIRS.protocol=='H0')&(PAIRS.metric=='top1')
    row=PAIRS[mask&(PAIRS['first']==first)&(PAIRS['second']==second)]
    if len(row):
        r=row.iloc[0]; wins=int(r['positive'])
    else:
        r=PAIRS[mask&(PAIRS['first']==second)&(PAIRS['second']==first)].iloc[0]; wins=int(r['negative'])
    return wins,float(r[pcol])
