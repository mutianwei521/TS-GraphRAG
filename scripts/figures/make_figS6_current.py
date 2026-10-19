"""All prespecified settings, paired accuracy differences from fixed default."""
import json
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from nature_style import SD, QA, figure, export
from pastel_style import apply_style, INK, PALETTE
apply_style()

d=pd.read_csv(SD/'sensitivity_by_seed.csv')
PROTOCOLS=['H0','H2','H4','H8']
groups=[('Reference',[('default','Fixed default')]),
 ('Graph aggregation',[(f'beta_{v}',f'Propagation strength β = {v:g}') for v in [0,.05,.3,.5,.75]]+
                       [(f'graph_steps_{v}',f'Propagation steps = {v}') for v in [2,3,5]]),
 ('Semantic and community weights',[(f'semantic_weight_multiplier_{v}',f'Joint weight multiplier = {v:g}') for v in [0,.5,1.5,2]]),
 ('Retrieval depth',[(f'top_k_{v}',f'Retrieval depth k = {v}') for v in [3,5,20,40]]),
 ('Weak / strong signal thresholds',[(f'signal_thresholds_{lo}_{hi}',f'Thresholds = {lo:g} / {hi:g} m') for lo,hi in [(.05,.5),(.2,2),(.05,1),(.2,1),(.1,.5),(.1,2)]]),
 ('Semantic similarity cutoff',[(f'semantic_cosine_threshold_{v}',f'Cosine cutoff ≥ {v:g}') for v in [.3,.5,.7,.85]]),
 ('Individual channel removal',[(f'leave_out_{c}',f'Remove {lab}') for c,lab in [('prior','sensor prior'),('centroid','centroid matching'),('rank','rank matching'),('fingerprint','fingerprint retrieval'),('semantic','semantic retrieval'),('community','community retrieval')]]),
 ('Alternative weighting',[('uniform_channel_weights','Uniform channel weights')])]
settings=[s for _,items in groups for s,_ in items]
assert len(settings)==34 and set(settings)==set(d.setting)
base=d[d.setting=='default'].set_index(['seed','protocol']).top1_pct
rows=[]; labels=[]; header_rows=[]; values=[]
for group,items in groups:
    header_rows.append((len(values),group)); values.append([np.nan]*4);labels.append(None)
    for setting,label in items:
        vv=[]
        for p in PROTOCOLS:
            ss=d[(d.setting==setting)&(d.protocol==p)].set_index('seed').top1_pct
            assert len(ss)==5 and ss.index.nunique()==5
            diffs=ss-base.xs(p,level='protocol')
            mu=float(diffs.mean());sd=float(diffs.std(ddof=1));vv.append(mu)
            rows.append(dict(setting=setting,label=label,protocol=p,n_noise_seeds=5,n_queries_each=540,delta_mean_pp=mu,delta_sd_pp=sd,default_mean_top1_pct=float(base.xs(p,level='protocol').mean())))
        values.append(vv);labels.append(label)
V=np.array(values);n=len(V)
assert np.isfinite(V).sum()==136
limlo=float(np.nanmin(V));limhi=float(np.nanmax(V))
fig=figure(230)
fig.text(.04,.975,'Prespecified sensitivity of the fusion engine',fontsize=9,weight='bold',va='top')
fig.text(.04,.952,'34 settings · 540 queries × 5 measurement-noise seeds · fixed default retained',fontsize=6.8,va='top')
fig.text(.49,.922,'Top-1 accuracy change from default (pp)',fontsize=7,va='top')
ax=fig.add_axes([.49,.125,.47,.745])
cmap=LinearSegmentedColormap.from_list('pastel_gain_loss',['#D3817D','#FFFEFC','#9C8BC2'])
norm=TwoSlopeNorm(vmin=limlo,vcenter=0,vmax=limhi)
mesh=ax.pcolormesh(np.arange(5),np.arange(n+1),np.ma.masked_invalid(V),cmap=cmap,norm=norm,edgecolors='white',linewidth=.5)
ax.set_xlim(0,4);ax.set_ylim(n,0)
defaults=[float(base.xs(p,level='protocol').mean()) for p in PROTOCOLS]
ax.set_xticks(np.arange(4)+.5,[f'{p}\nDefault {v:.2f}%' for p,v in zip(PROTOCOLS,defaults)])
ax.xaxis.tick_top();ax.tick_params(axis='x',length=0,pad=6,labelsize=6.5)
ax.set_yticks([])
for spine in ax.spines.values():spine.set_visible(False)
for j,(label,vals) in enumerate(zip(labels,V)):
    if label is None:continue
    ax.text(-.11,j+.5,label,ha='right',va='center',fontsize=6.3,clip_on=False)
    for k,v in enumerate(vals):
        s=f'{v:+.1f}' if abs(v)>.049 else '0.0'
        ax.text(k+.5,j+.5,s.replace('-','−'),ha='center',va='center',fontsize=6.1,color=INK)
for j,group in header_rows:
    ax.text(-3.82,j+.62,group,ha='left',va='center',fontsize=6.5,weight='bold',clip_on=False)
    ax.plot([-3.82,4],[j+.95,j+.95],color='#D9D9D9',lw=.45,clip_on=False)
cax=fig.add_axes([.49,.075,.47,.014])
cb=fig.colorbar(mesh,cax=cax,orientation='horizontal')
cb.set_ticks([-50,-25,0,5,10]);cb.ax.tick_params(length=2,width=.45,labelsize=6)
cb.outline.set_visible(False)
fig.text(.49,.06,'Decrease',fontsize=6.3,ha='left',va='top')
fig.text(.96,.06,'Increase',fontsize=6.3,ha='right',va='top')
fig.text(.04,.027,'Cells: mean paired difference across five seeds. All settings shown; no test-based selection.',fontsize=6.5,va='bottom')
pd.DataFrame(rows).to_csv(SD/'FigS6_engine_sensitivity_plotted_values.csv',index=False)
(QA/'FigS6_data_checks.json').write_text(json.dumps({'settings':34,'protocols':PROTOCOLS,'numeric_cells':136,'noise_seeds':5,'queries_each':540,'mean_definition':'paired difference from seed-matched default','error_definition':'seed SD retained in source data; not encoded in heatmap','before_rows':680,'after_rows':680,'excluded_settings':[],'color_range_pp':[limlo,limhi],'color_clipping':False,'default_preserved':True},indent=2))
export(fig,'FigS6_engine_sensitivity')
