from pathlib import Path
import json,sys,hashlib
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.text import Text
from _paths import INPUT, OUTPUT
OUT=OUTPUT; H=INPUT/'hydraulic_robustness'

def label(c):
 f,v=c['family'],c['value']
 return {'control':'Nominal','roughness':f'Roughness CV {v:.0%}','reservoir_head':f'Source head ±{v:g} m','pump_speed':f'Pump speed ±{v:.0%}','spatial_demand':f'Spatial demand CV {v:.0%}','spatiotemporal_demand':f'Space–time demand CV {v:.0%}','phase_hours':f'Demand phase ±{v:g} h','uniform_demand':f'Demand multiplier {v:g}','joint':'Joint mild' if v==.5 else 'Joint moderate'}[f]
def pm(m,sd):
 if pd.isna(m):return 'n.a.'
 if pd.isna(sd):return f'{m:.1f} (s.d. n.a.)'
 return f'{m:.1f} ± {sd:.1f}'

def main():
 audit=json.loads((H.parent/'strict_engine/cross_audit_hydraulic_final.json').read_text(encoding='utf8'))
 assert audit['status']=='passed' and audit['error106_count']==0 and audit['error307_count']==0
 # Summary replay checks the included audited aggregate; full hydraulic rerun is separate.
 for name,expected in [('summary.csv', audit['source_hashes']['summary.csv'])]:
  assert hashlib.sha256((H/name).read_bytes()).hexdigest()==expected,name
 s=pd.read_csv(H/'summary.csv');seed=pd.read_csv(H/'seed_summary.csv');design=json.loads((H/'design.json').read_text(encoding='utf8'));conds=design['scenarios']
 def row(c,b='matched',policy='nominal_day'):
  f=s[(s.scenario==c)&(s.baseline==b)&(s.library_policy==policy)&(s.method=='projection')];assert len(f)==1;return f.iloc[0]
 def nvalid(c,b='matched',policy='nominal_day'):
  f=seed[(seed.scenario==c)&(seed.baseline==b)&(seed.library_policy==policy)&(seed.method=='projection')];return int((f.n_admissible>0).sum())
 plt.rcParams.update({'font.family':'Arial','font.size':7,'axes.labelsize':7,'xtick.labelsize':6.5,'ytick.labelsize':6.5,'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.6,'pdf.fonttype':42,'svg.fonttype':'none','text.color':'#3E3E3E','axes.labelcolor':'#3E3E3E'})
 width_in,height_in=183/25.4,180/25.4
 fig=plt.figure(figsize=(width_in,height_in));axes=[fig.add_axes(p) for p in [[.25,.10,.24,.76],[.565,.10,.24,.76],[.875,.10,.10,.76]]]
 colors=['#9DD9CE','#B7ADD6'];fields=['top1_all_pct','top1_admissible_pct']
 for a,field in zip(axes[:2],fields):
  for i,c in enumerate(conds):
   for j,b in enumerate(['matched','stale_nominal']):
    r=row(c['name'],b);v=r[field+'_mean'];sd=r[field+'_sd']
    if np.isfinite(v):
     a.barh(i+(j-.5)*.33,v,height=.30,color=colors[j],edgecolor='#ABB3B8',linewidth=.3,zorder=2)
     if np.isfinite(sd):a.errorbar(v,i+(j-.5)*.33,xerr=sd,fmt='none',ecolor='#6E7479',elinewidth=.45,capsize=1,zorder=3)
    else:a.text(3,i,'n.a.',fontsize=6,va='center')
  a.set_xlim(0,105);a.set_xticks([0,50,100]);a.set_xlabel('Queries (%)');a.set_yticks(range(len(conds)))
 axes[0].set_xlim(-15,105)
 axes[0].axvline(0,color='#6E7479',lw=.45,zorder=1)
 axes[0].set_yticklabels([label(c) for c in conds]);axes[1].set_yticklabels([])
 for i,c in enumerate(conds):
  r=row(c['name']);v=100-r.failure_pct_mean
  axes[2].barh(i,v,height=.64,color='#AAC4E0',edgecolor='#ABB3B8',linewidth=.3)
  axes[2].text(min(v+3,102),i,f'{nvalid(c["name"])}',ha='left',va='center',fontsize=6,clip_on=False)
 axes[2].set_xlim(0,105);axes[2].set_xticks([0,100]);axes[2].set_yticks([]);axes[2].set_xlabel('Valid (%)')
 for a in axes:
  a.set_ylim(len(conds)-.5,-.5);a.tick_params(axis='y',length=0,pad=5)
  for i in [0,3,6,9,12,15,18,23]:a.axhline(i+.5,color='#E1E5E9',lw=.5,zorder=0)
 for x,l,t in [(.25,'a','All planned queries'),(.565,'b','Admissible queries only'),(.875,'c','Validity')]:
  fig.text(x-.025,.887,l,weight='bold',fontsize=8);fig.text(x,.887,t,fontsize=7)
 fig.text(.04,.972,'Hydraulic and demand mismatch',fontsize=9,weight='bold')
 fig.text(.04,.944,'Frozen day-normal library · 60 unseen-node queries × 5 noise/perturbation realizations',fontsize=7)
 fig.legend([Patch(fc=c,ec='#ABB3B8',lw=.3) for c in colors],['Matched operating baseline','Stale nominal baseline'],loc='upper center',bbox_to_anchor=(.56,.93),ncol=2,frameon=False,fontsize=7)
 fig.text(.04,.037,'Means ± sample s.d.; invalid queries count as failures in a. Numbers in c: seeds with admissible queries (0–5).',fontsize=6.3)
 fig.canvas.draw();ren=fig.canvas.get_renderer();w,h=fig.canvas.get_width_height();overflow=[]
 for t in fig.findobj(match=Text):
  if t.get_visible() and t.get_text():
   bb=t.get_window_extent(ren)
   if bb.x0<-.5 or bb.x1>w+.5 or bb.y0<-.5 or bb.y1>h+.5:overflow.append(t.get_text())
 assert not overflow,overflow
 stem='FigS3_hydraulic_mismatch'
 fig.savefig(OUT/'figures/FigS3_hydraulic_mismatch.png',dpi=600)
 fig.savefig(OUT/'figures/FigS3_hydraulic_mismatch.pdf',dpi=600)
 fig.savefig(OUT/'figures/FigS3_hydraulic_mismatch.svg',dpi=600)
 cap=('Fig. S3 | Hydraulic and demand mismatch with a frozen nominal dictionary. All 26 prespecified conditions are shown for projection-residual retrieval using the primary day-normal library (hydraulically screened templates; all 30 query junctions excluded). a, Correct and hydraulically admissible queries as a percentage of all 60 planned queries in each realization. Invalid cases remain in the denominator and count as failures. b, Top-1 accuracy conditional on admissible queries; n.a. denotes no admissible queries. c, Mean admissibility rate; the small numbers report the number of seeds with at least one admissible query. Bars in a,b show mean ± sample s.d.; a uses five realizations, whereas b uses only realizations with admissible queries and can have fewer than five. Matched subtracts the no-leak baseline under the same perturbed model and demand; stale subtracts the nominal day-normal baseline. Admissibility requires finite, converged pressures in [0, 200] m at every junction and hourly report, with no hydraulic report warnings. The display includes negative lower SD endpoints; these are not negative probabilities. Negative pump-suction auxiliary-node pressures can trigger this conservative criterion without negative service-node pressures; a separate service-node diagnostic is supplied. Thus reduced success in a can reflect physical invalidity as well as retrieval error. Demand CV denotes the target before total-demand normalization. Perturbation distributions and exact amplitudes are defined in Supplementary Note 1.10. Deterministic control hydraulics are reused with five independent noise realizations; random perturbations vary both hydraulics and noise. These are synthetic tests on daily-average pressure features, not measured field or real-time SCADA performance. All methods, library policies, valid-query counts and raw predictions are supplied in Source Data.')
 (OUT/'figures/FigS3_caption.txt').write_text(cap,encoding='utf8')
 input_hashes={n:hashlib.sha256((H/n).read_bytes()).hexdigest() for n in ['summary.csv','seed_summary.csv','design.json']}
 (OUT/'qa/FigS3_checks.json').write_text(json.dumps({'conditions':len(conds),'width_mm':183,'height_mm':180,'overflow':overflow,'primary_library':'nominal_day','mean_sd_unit':'seed realization','no_condition_omitted':True,'input_sha256':input_hashes},indent=2),encoding='utf8')
 tables=[]
 for policy,part in [('nominal_day','a, Primary day-normal library'),('all_nominal_periods','b, Secondary three-state union library')]:
  rows=[]
  for c in conds:
   a=row(c['name'],'matched',policy);b=row(c['name'],'stale_nominal',policy)
   rows.append([label(c),pm(100-a.failure_pct_mean,a.failure_pct_sd),pm(a.top1_all_pct_mean,a.top1_all_pct_sd),pm(b.top1_all_pct_mean,b.top1_all_pct_sd),pm(a.top1_admissible_pct_mean,a.top1_admissible_pct_sd)+f' (n={nvalid(c["name"],policy=policy)})'])
  tables.extend([{'kind':'p','text':f'Table S13 | {part}. Projection-residual results for all 26 prespecified conditions. All percentages are mean ± sample s.d. across five seeds except conditional accuracy, for which n gives the number of seeds containing admissible queries. The all-query success endpoint requires both hydraulic admissibility and a correct Top-1; no failed query is removed from its denominator. The matched and stale baselines share the same hydraulic validity. Complete valid-only stale outcomes, comparison methods and privileged known-period controls are in Source Data.'},{'kind':'table','headers':['Condition','Admissible (%)','Matched success (%)','Stale success (%)','Matched valid-only (%)'],'rows':rows,'widths':[1.6,1.1,1.2,1.2,1.3]}])
 (OUT/'source_data/TableS13_display.json').write_text(json.dumps(tables,ensure_ascii=False,indent=2),encoding='utf8')
 print('FigS3 and Table S13 rebuilt from the audited frozen summaries')

if __name__=='__main__':main()
