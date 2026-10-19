"""Fig. 5: all 15 allocation draws, using the supplied pastel reference palette.

Structural/style adaptation only: frozen observations and statistical contrasts
are unchanged. Shared export: Fig5_corpus_allocation.png/.pdf/.svg, dpi=600.
"""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from nature_style import *
from allocation_data import *
from pastel_style import BLUE, RED, ORANGE, PURPLE, TEAL, YELLOW, GREY, INK, darker, apply_style

plt.rcParams.update({'font.sans-serif':['Arial'],'font.size':7,
                     'svg.fonttype':'none','pdf.fonttype':42})
apply_style()
WIDTH_MM=183
fig=figure(163)
gs=fig.add_gridspec(2,2,left=.09,right=.98,bottom=.105,top=.935,
                    wspace=.31,hspace=.47)
a,b,c,d=[fig.add_subplot(g) for g in gs]
counts={}; plotted=[]; paired={}; x=np.arange(5)


def series(ax,group,metric,color,offset=0,width=.65):
    for i,design in enumerate(DES):
        mean,sd,vals=stat(group,design,metric)
        face=color[i] if isinstance(color,list) else color
        bar=ax.bar(i+offset,mean,width,color=face,edgecolor='#888888',lw=.35,
                   hatch='/' if design=='A_prime' else None,zorder=2)
        ax.errorbar(i+offset,mean,yerr=sd,fmt='none',ecolor=INK,
                    elinewidth=.65,capsize=2.5,capthick=.65,zorder=4)
        # All fifteen draws are displayed, with fixed horizontal displacement.
        ax.scatter(i+offset+np.linspace(-.32,.32,15)*width,vals,
                   s=7,fc='white',ec='#777777',lw=.3,zorder=5)
        counts[f'{group}/{metric}/{design}']=len(vals)
        for draw,value in enumerate(vals,1):
            plotted.append({'panel':ax.get_label(),'group':group,'metric':metric,
                            'design':design,'draw':draw,'value':float(value)})


def design_axis(ax,ylim):
    ax.set_xticks(x,LABELS); ax.set_xlim(-.65,4.65); ax.set_ylim(*ylim)
    clean(ax,grid=True)


for ax,name in zip([a,b,c,d],'abcd'): ax.set_label(name)
series(a,'G1','top1',BLUE,-.185,.34)
series(a,'G2','top1',RED,.185,.34)
design_axis(a,(0,108)); a.set_yticks([0,20,40,60,80,100])
a.set_ylabel('Top-1 accuracy (%)'); a.set_title('Location coverage',loc='left',pad=8)
a.legend(handles=[Patch(fc=BLUE,ec='#888888',lw=.35,label='G1: represented'),
                  Patch(fc=RED,ec='#888888',lw=.35,label='G2: unseen')],
         loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,columnspacing=.9,handlelength=1.4)

# Solid paired bars preserve every derived contrast; x is draw identity, not
# a continuous corpus-size axis. Dotted lines give the two paired medians.
draws=np.arange(1,16)
for second,color,offset in [('A',PURPLE,-.19),('B',TEAL,.19)]:
    delta=values('G2','C90','top1')-values('G2',second,'top1')
    wins,p=pair('C90',second); med=float(np.median(delta))
    b.bar(draws+offset,delta,.35,color=color,edgecolor='#888888',lw=.35,zorder=3,
          label=f'C90 − {second}: Holm $P$ = {p:.2g}')
    b.axhline(med,color=darker(color),lw=.8,ls=(0,(3,2)),zorder=4)
    paired[f'C90_minus_{second}']={'differences_pp':delta.tolist(),
                                   'positive_draws':wins,'holm_p':p,'median_pp':med}
    counts[f'paired/C90-{second}']=len(delta)
b.axhline(0,color=INK,lw=.6,zorder=2)
b.set_xticks(draws,[str(i) for i in draws]); b.tick_params(axis='x',labelsize=6)
b.set_xlim(.35,15.65); b.set_ylim(-12,36); b.set_yticks([-10,0,10,20,30])
b.set_ylabel('Paired Top-1 difference (pp)'); b.set_xlabel('Draw')
b.set_title('Unseen-location contrasts',loc='left',pad=8)
b.legend(loc='upper left',fontsize=6.25,handlelength=1.6,borderaxespad=.25,labelspacing=.35)
clean(b,grid=True)

series(c,'G2','top3',YELLOW,-.185,.34)
series(c,'G2','mrr',PURPLE,.185,.34)
design_axis(c,(0,108)); c.set_yticks([0,20,40,60,80,100])
c.set_ylabel('Score (%)'); c.set_title('Candidate ranking',loc='left',pad=8)
c.legend(handles=[Patch(fc=YELLOW,ec='#888888',lw=.35,label='Top-3'),
                  Patch(fc=PURPLE,ec='#888888',lw=.35,label='MRR@5 × 100')],
         loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,columnspacing=1,handlelength=1.4)

series(d,'G2','med_m',[BLUE,PURPLE,TEAL,YELLOW,ORANGE],width=.62)
design_axis(d,(0,320)); d.set_yticks([0,50,100,150,200,250,300])
d.set_ylabel('Distance (coordinate units)'); d.set_title('Spatial error',loc='left',pad=8)
for i,design in enumerate(DES):
    m,s,v=stat('G2',design,'med_m')
    d.text(i,max(m+s,v.max())+10,f'{m:.0f}',ha='center',fontsize=6.5)

for ax,l in zip([a,b,c,d],'abcd'): letter(ax,l,x=-.18,y=1.14)
assert len(plotted)==375
assert all(n==15 for n in counts.values())
assert np.allclose(paired['C90_minus_A']['differences_pp'],
                   pd.read_csv(SD/'Fig5b_paired_differences.csv').C90_minus_A_pp)
assert np.allclose(paired['C90_minus_B']['differences_pp'],
                   pd.read_csv(SD/'Fig5b_paired_differences.csv').C90_minus_B_pp)
pd.DataFrame(plotted).to_csv(SD/'Fig5_allocation_plotted_observations.csv',index=False)
(QA/'Fig5_data_checks.json').write_text(json.dumps({
    'observations_per_series':counts,'plotted_observations':375,
    'derived_paired_differences':30,'paired':paired,'excluded_observations':0,
    'statistics':'arithmetic mean and sample SD, ddof=1; paired lines are medians',
    'source_checks':'draw means/SD match frozen summaries <0.06; paired bars match frozen CSV',
},indent=2),encoding='utf-8')
fig.canvas.draw()
texts=[obj for obj in fig.findobj(match=matplotlib.text.Text) if obj.get_visible() and obj.get_text().strip()]
renderer=fig.canvas.get_renderer(); overlaps=[]
for i,left in enumerate(texts):
    for right in texts[i+1:]:
        if left.get_window_extent(renderer).expanded(.98,.90).overlaps(right.get_window_extent(renderer).expanded(.98,.90)):
            overlaps.append([left.get_text(),right.get_text()])
(QA/'Fig5_text_audit.json').write_text(json.dumps({'overlapping_text_pairs':overlaps},indent=2),encoding='utf-8')
print('Text overlaps:',overlaps)
export(fig,'Fig5_corpus_allocation')
