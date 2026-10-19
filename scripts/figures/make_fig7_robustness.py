"""Fig. 7: all 52 robustness outcomes, using the supplied pastel palette.

No outcomes are sampled or dropped. These are single-realization results:
there are no invented SD bars. Fig7_robustness_networks.png/.pdf/.svg, dpi=600.
"""
import json,csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from nature_style import *
from pastel_style import BLUE, RED, ORANGE, PURPLE, TEAL, YELLOW, GREY, INK, apply_style

plt.rcParams.update({'font.sans-serif':['Arial'],'font.size':7,
                     'svg.fonttype':'none','pdf.fonttype':42})
apply_style()
WIDTH_MM=183
exp3=json.loads((ROOT/'exp3_results.json').read_text())
a2=json.loads((ROOT/'expA2_results.json').read_text())
a1b=json.loads((ROOT/'expA1b_results.json').read_text())
a1v2=json.loads((ROOT/'expA1_v2_results.json').read_text())
fig=figure(165)
gs=fig.add_gridspec(2,2,left=.085,right=.98,bottom=.12,top=.935,
                    wspace=.49,hspace=.57)
a,b,c,d=[fig.add_subplot(g) for g in gs]
all_values={}; plot_records=[]


def retain(key,values):
    vals=np.asarray(values,dtype=float)
    assert np.isfinite(vals).all() and ((vals>=0)&(vals<=100)).all()
    all_values[key]=vals.tolist()
    for i,value in enumerate(vals): plot_records.append({'series':key,'index':i,'value_percent':float(value)})
    return vals


alphas=['0.5','1.0','1.5']; x=np.arange(3)
spec=[('Cd','part',BLUE,-.30,None,'Demand: partition'),
      ('Ce','part',ORANGE,-.10,None,'Emitter: partition'),
      ('Cd','node',BLUE,.10,'/','Demand: junction'),
      ('Ce','node',ORANGE,.30,'/','Emitter: junction')]
for corpus,metric,color,offset,hatch,label in spec:
    vals=retain(f'a/{corpus}/{metric}',[100*exp3['q2'][f'a{p}_{corpus}'][metric] for p in alphas])
    a.bar(x+offset,vals,.185,color=color,edgecolor='#888888',lw=.35,hatch=hatch,label=label,zorder=3)
ctrl=100*exp3['q2']['control']['part']
a.axhline(ctrl,color=INK,lw=.65,ls=(0,(4,2)),zorder=4)
a.set_xticks(x,alphas); a.set_xlim(-.6,2.6); a.set_ylim(0,105)
a.set_ylabel('Accuracy (%)'); a.set_xlabel('Emitter exponent α')
a.set_title('Leak-model form',loc='left',pad=8)
a.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2,
         columnspacing=.7,handlelength=1.4,fontsize=6.1,labelspacing=.35)

order=['baseline, 30 deployed sensors','constant bias (control: cancels)',
       'drift sd=0.05 m','drift sd=0.15 m','drift sd=0.30 m','missing 10%',
       'missing 30%','1 sensor stuck','3 sensors stuck','5% impulsive spikes (1 m)',
       'bad day: drift .15 + 10% miss + 2 stuck + spikes']
labels=['Baseline','Constant bias','Drift 0.05 m','Drift 0.15 m','Drift 0.30 m',
        'Missing 10%','Missing 30%','1 stuck','3 stuck','Spikes 5%','Combined']
y=np.arange(len(order))[::-1]
v1=retain('b/projection',[100*a2['results'][k]['L2']['recall'] for k in order])
v2=retain('b/trimmed',[100*a2['results'][k]['trimmed']['recall'] for k in order])
b.axhspan(-.48,.48,color='#FFF8E9',zorder=0)
b.barh(y+.175,v1,.31,color=PURPLE,edgecolor='#888888',lw=.35,label='Projection',zorder=3)
b.barh(y-.175,v2,.31,color=TEAL,edgecolor='#888888',lw=.35,label='Trimmed',zorder=3)
b.set_yticks(y,labels); b.tick_params(axis='y',labelsize=6.3)
b.set_ylim(-.7,10.7); b.set_xlim(0,100)
b.set_xlabel('Class recall (%)'); b.set_title('Sensor degradation',loc='left',pad=8)
b.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2,columnspacing=.8,handlelength=1.4)

v1=retain('c/exhaustive',[100*a1v2['results']['balanced_k1']['oracle_class_recall'],
                         100*a1b['D1']['2']['exhaustive'],100*a1b['D1']['3']['exhaustive']])
v2=retain('c/greedy',[100*a1v2['results']['balanced_k1']['oracle_class_recall'],
                     100*a1b['D1']['2']['greedy'],100*a1b['D1']['3']['greedy']])
for off,vals,color,label in [(-.18,v1,BLUE,'Exhaustive'),(.18,v2,ORANGE,'Greedy')]:
    c.bar(x+off,vals,.33,color=color,edgecolor='#888888',lw=.35,label=label,zorder=3)
    for xx,yy in zip(x+off,vals): c.text(xx,yy+2.4,f'{yy:.1f}',ha='center',fontsize=6.5)
c.set_xticks(x,['1','2','3']); c.set_xlim(-.6,2.6); c.set_ylim(0,105)
c.set_xlabel('Simultaneous leaks'); c.set_ylabel('Classes recovered (%)')
c.set_title('Multiple leaks',loc='left',pad=8)
c.legend(loc='upper center',bbox_to_anchor=(.5,-.19),ncol=2,columnspacing=.8,handlelength=1.4)

rows={}
with open(ROOT/'0829'/'SI_new_tables'/'TableS11.txt',encoding='utf-8') as f:
    for row in csv.DictReader(f,delimiter='\t'):
        if row['Network']=='EXA7': continue
        key=(row['Network'],'G1' if row['Query group'].startswith('query node in') else 'G2',
             row['Corpus design'][:1],row['Protocol (what is deleted)'].split()[0])
        rows[key]=float(row['Top-1 (%)'])
groups=[('City D','A'),('City D','B'),('Haining','A'),('Haining','B')]
for off,group,protocol,color,label in [(-.25,'G1','H0',BLUE,'G1/H0'),
        (0,'G1','H2',BLUE,'G1/H2'),(.25,'G2','H0',RED,'G2')]:
    vals=retain(f'd/{group}/{protocol}',[rows[(net,group,design,protocol)] for net,design in groups])
    d.bar(np.arange(4)+off,vals,.23,color=color,edgecolor='#888888',lw=.35,
          hatch='/' if protocol=='H2' else None,label=label,zorder=3)
d.set_xticks(range(4),['A','B','A','B']); d.set_xlim(-.6,3.6); d.set_ylim(0,105)
d.axvline(1.5,color=GREY,lw=.65,ls=(0,(3,3)))
d.text(.5,-.15,'City D',transform=d.get_xaxis_transform(),ha='center',fontsize=6.5)
d.text(2.5,-.15,'Haining',transform=d.get_xaxis_transform(),ha='center',fontsize=6.5)
d.set_ylabel('Top-1 accuracy (%)'); d.set_title('Additional networks',loc='left',pad=8)
d.legend(loc='upper center',bbox_to_anchor=(.5,-.25),ncol=3,columnspacing=.75,handlelength=1.25)

for ax,label in zip([a,b,c,d],'abcd'):
    if ax is not b: ax.set_yticks([0,20,40,60,80,100])
    clean(ax,grid=ax is not b)
    if ax is b:
        ax.set_axisbelow(True); ax.grid(axis='x',color='#E9EDF0',lw=.4)
    letter(ax,label,x=-.19 if ax is not b else -.36,y=1.14)
assert sum(len(v) for v in all_values.values())==52
# The source values agree exactly with the previous complete 52-value audit.
prior=json.loads((ROOT/'0829'/'figure_redesign_20260912'/'qa'/'Fig7_data_checks.json').read_text())
assert set(prior['values'])==set(all_values)
for key in all_values: assert np.array_equal(np.array(all_values[key]),np.array(prior['values'][key]))
pd.DataFrame(plot_records).to_csv(SD/'Fig7_robustness_plotted_values.csv',index=False)
(QA/'Fig7_data_checks.json').write_text(json.dumps({
    'values':all_values,'count':52,'excluded_observations':0,
    'control_reference_percent':ctrl,'source_comparison':'all 52 values exactly match previous plotted values',
    'replicate_uncertainty':'single realization; no SD, interval, or replicate error bars',
},indent=2),encoding='utf-8')
fig.canvas.draw()
texts=[obj for obj in fig.findobj(match=matplotlib.text.Text) if obj.get_visible() and obj.get_text().strip()]
renderer=fig.canvas.get_renderer(); overlaps=[]
for i,left in enumerate(texts):
    for right in texts[i+1:]:
        if left.get_window_extent(renderer).expanded(.98,.90).overlaps(right.get_window_extent(renderer).expanded(.98,.90)):
            overlaps.append([left.get_text(),right.get_text()])
(QA/'Fig7_text_audit.json').write_text(json.dumps({'overlapping_text_pairs':overlaps},indent=2),encoding='utf-8')
print('Text overlaps:',overlaps)
export(fig,'Fig7_robustness_networks')
