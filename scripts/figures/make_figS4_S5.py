"""Audited supplementary design controls; actual experiment tables only."""
from pathlib import Path
import sys,json,shutil
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.text import Text

from _paths import INPUT, OUTPUT, FIGDIR, QA
OUT=INPUT/'partition_sensor'
from pastel_style import PALETTE,apply_style,darker,INK
apply_style()
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
 'font.size':7,'axes.labelsize':7,'axes.titlesize':8,'axes.titleweight':'normal',
 'xtick.labelsize':6.5,'ytick.labelsize':6.5,'legend.fontsize':6.5,
 'axes.spines.top':False,'axes.spines.right':False,'legend.frameon':False,
 'svg.fonttype':'none','pdf.fonttype':42,'figure.facecolor':'white','axes.facecolor':'white',
 'xtick.major.size':2.5,'ytick.major.size':2.5,'savefig.dpi':600})
FIG=FIGDIR; DELIVERY=FIGDIR
for d in [FIG,QA,DELIVERY]: d.mkdir(exist_ok=True,parents=True)
RUN=pd.read_csv(OUT/'run_level_summary.csv')
STRUCT=pd.read_csv(OUT/'partition_structure_summary.csv')
ARI=pd.read_csv(OUT/'partition_stability.csv')
PART=['pressure_leiden','topology_leiden','train_hydraulic_ward']
SENSOR=['random','topological_maximin','train_sensitivity_doptimal']
PARTLABEL=['Pressure Leiden','Topology Leiden','Hydraulic Ward']
SENSLABEL=['Random','Topology maximin','Response D-optimal']
COL=[PALETTE['purple'],PALETTE['blue'],PALETTE['mint']]
SCOL=[PALETTE['blue'],PALETTE['mint'],PALETTE['purple']]
MARK=['o','s','^']; KS=[5,10,15,20,30,40]; BS=[10,15,20,30,45,60]
data=[]

def record(fig,panel,method,x,values,metric,unit):
    data.append(dict(figure=fig,panel=panel,method=method,x=x,metric=metric,unit=unit,
                     mean=float(np.mean(values)),sd=float(np.std(values,ddof=1)),n=len(values),
                     values=';'.join(f'{v:.12g}' for v in values)))

def setup(ax,letter,title):
    ax.set_title(title,loc='left',pad=8)
    ax.text(-.16,1.11,letter,transform=ax.transAxes,fontsize=8,fontweight='bold',va='top',color=INK)
    ax.tick_params(pad=2); ax.grid(axis='y',color='#EBEDF0',lw=.4,zorder=0)
    ax.set_axisbelow(True)

def line(ax,subset,xcol,ycol,method,color,marker,fig,panel,metric,unit,scale=1):
    means=[]; sds=[]; xs=[]
    for x,group in subset.groupby(xcol):
        vals=group[ycol].to_numpy()*scale
        record(fig,panel,method,x,vals,metric,unit)
        xs.append(x); means.append(vals.mean()); sds.append(vals.std(ddof=1))
    ax.errorbar(xs,means,yerr=sds,color=darker(color,.72),marker=marker,markersize=3.5,
                markerfacecolor=color,markeredgecolor=darker(color,.65),markeredgewidth=.5,
                linewidth=1.15,elinewidth=.7,capsize=2,capthick=.7,zorder=3)

def export(fig,name):
    fig.canvas.draw(); renderer=fig.canvas.get_renderer(); width,height=fig.canvas.get_width_height()
    texts=[]; overflow=[]
    for item in fig.findobj(match=Text):
        if not item.get_visible() or not item.get_text().strip(): continue
        box=item.get_window_extent(renderer)
        texts.append(dict(text=item.get_text(),size_pt=item.get_fontsize(),bbox=[box.x0,box.y0,box.x1,box.y1]))
        if box.x0<-.5 or box.y0<-.5 or box.x1>width+.5 or box.y1>height+.5: overflow.append(item.get_text())
    report=dict(figure=name,width_mm=fig.get_figwidth()*25.4,height_mm=fig.get_figheight()*25.4,
                min_font_pt=min(t['size_pt'] for t in texts),overflow=overflow,text=texts)
    (QA/f'{name}.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert not overflow,overflow
    fig.savefig(FIG/f'{name}.png',dpi=600,facecolor='white',bbox_inches=None)
    fig.savefig(FIG/f'{name}.pdf',facecolor='white',bbox_inches=None)
    fig.savefig(FIG/f'{name}.svg',facecolor='white',bbox_inches=None)
    fig.savefig(QA/f'{name}_preview.png',dpi=300,facecolor='white',bbox_inches=None)
    plt.close(fig); print(name,'exported',report['width_mm'],report['height_mm'])

def partition():
    name='FigS4_partition_controls'; f=plt.figure(figsize=(7.204724409448819,4.803149606299213),dpi=300)
    axes=[f.add_axes(p) for p in [[.10,.585,.36,.30],[.60,.585,.36,.30],[.10,.115,.36,.30],[.60,.115,.36,.30]]]
    titles=['Regional localization','Candidate pipe extent','Partition boundaries','Assignment stability']
    for ax,letter,title in zip(axes,'abcd',titles):
        setup(ax,letter,title); ax.set_xticks(KS); ax.set_xlim(3,42); ax.set_xlabel('Number of regions, K')
    for i,m in enumerate(PART):
        r=RUN[(RUN.experiment=='partition')&(RUN.method==m)]
        st=STRUCT[STRUCT.method==m]; ar=ARI[ARI.method==m]
        line(axes[0],r,'k','top1',m,COL[i],MARK[i],name,'a','regional_top1','percent',100)
        line(axes[1],r,'k','mean_inspection_pipe_m',m,COL[i],MARK[i],name,'b','mean_candidate_pipe_length','km',.001)
        line(axes[2],st,'k','boundary_pipe_m',m,COL[i],MARK[i],name,'c','network_boundary_pipe_length','km',.001)
        line(axes[3],ar,'k','ari',m,COL[i],MARK[i],name,'d','pairwise_adjusted_rand_index','unitless')
    axes[0].set_ylabel('Top-1 accuracy (%)'); axes[0].set_ylim(50,103); axes[0].set_yticks([50,60,70,80,90,100])
    axes[1].set_ylabel('Mean candidate pipe length (km)'); axes[1].set_ylim(0,20); axes[1].set_yticks([0,5,10,15,20])
    axes[2].set_ylabel('Boundary pipe length (km)'); axes[2].set_ylim(0,12); axes[2].set_yticks([0,3,6,9,12])
    axes[3].set_ylabel('Adjusted Rand index'); axes[3].set_ylim(.4,1.035); axes[3].set_yticks([.4,.6,.8,1])
    handles=[Line2D([0],[0],color=darker(c,.72),marker=m,ms=4,mfc=c,mec=darker(c,.65),lw=1.15) for c,m in zip(COL,MARK)]
    f.legend(handles,PARTLABEL,loc='upper center',bbox_to_anchor=(.53,.985),ncol=3,columnspacing=2.5,handlelength=2)
    export(f,name)

def heatmap(ax,cax,metric,label,cmap,lo,hi,fig,panel):
    noise=[0,.01,.02,.05,.1]; drops=[0,.1,.25,.5]
    values=np.zeros((5,4)); sd=np.zeros_like(values)
    sub=RUN[RUN.experiment=='noise_dropout']
    factor=100 if metric=='top1' else 1
    for i,sigma in enumerate(noise):
        for j,drop in enumerate(drops):
            part=sub[(sub.sigma_m==sigma)&(sub.dropout==drop)]
            vals=part[metric].to_numpy()*factor
            assert len(vals)==5
            values[i,j]=vals.mean(); sd[i,j]=vals.std(ddof=1)
            record(fig,panel,f'sigma={sigma:g}',int(part.budget.iloc[0]),vals,metric,'percent' if factor==100 else 'm')
    im=ax.imshow(values,cmap=cmap,vmin=lo,vmax=hi,aspect='auto',interpolation='nearest')
    for i in range(5):
        for j in range(4):
            txt=f'{values[i,j]:.1f}±{sd[i,j]:.1f}' if factor==100 else f'{values[i,j]:.0f}±{sd[i,j]:.0f}'
            ax.text(j,i,txt,ha='center',va='center',fontsize=6.3,color='#333333')
    ax.set_xticks(range(4),[30,27,22,15]); ax.set_yticks(range(5),['0','0.01','0.02','0.05','0.10'])
    ax.set_xlabel('Available sensors'); ax.set_ylabel('Noise SD (m)'); ax.grid(False)
    ax.set_xticks(np.arange(-.5,4,1),minor=True); ax.set_yticks(np.arange(-.5,5,1),minor=True)
    ax.grid(which='minor',color='white',linewidth=1); ax.tick_params(which='minor',length=0)
    for spine in ax.spines.values(): spine.set_visible(False)
    cb=plt.colorbar(im,cax=cax); cb.ax.tick_params(labelsize=6,length=2,pad=2); cb.outline.set_linewidth(.4)
    cb.set_label(label,size=6.5,labelpad=3)

def sensor():
    name='FigS5_sensor_controls'; f=plt.figure(figsize=(7.204724409448819,5.196850393700787),dpi=300)
    axes=[f.add_axes(p) for p in [[.105,.585,.35,.29],[.60,.585,.35,.29],[.105,.105,.31,.295],[.60,.105,.31,.295]]]
    caxes=[f.add_axes([.43,.105,.011,.295]),f.add_axes([.925,.105,.011,.295])]
    for ax,letter,title in zip(axes,'abcd',['Sensor budget','Physical localization error','Noise and sensor loss: accuracy','Noise and sensor loss: distance']): setup(ax,letter,title)
    for ax in axes[:2]: ax.set_xticks(BS); ax.set_xlim(7,63); ax.set_xlabel('Number of sensors')
    for i,m in enumerate(SENSOR):
        sub=RUN[(RUN.experiment=='sensor')&(RUN.method==m)]
        line(axes[0],sub,'budget','top1',m,SCOL[i],MARK[i],name,'a','regional_top1','percent',100)
        line(axes[1],sub,'budget','median_distance_m',m,SCOL[i],MARK[i],name,'b','run_median_pipe_distance','m')
    fixed=RUN[(RUN.experiment=='sensor')&(RUN.method=='historical_deployed')]
    for j,(metric,scale,unit) in enumerate([('top1',100,'percent'),('median_distance_m',1,'m')]):
        line(axes[j],fixed,'budget',metric,'historical_deployed',PALETTE['peach'],'D',name,'ab'[j],metric,unit,scale)
    axes[0].set_ylabel('Top-1 accuracy (%)'); axes[0].set_ylim(40,100); axes[0].set_yticks([40,50,60,70,80,90,100])
    axes[1].set_ylabel('Median pipe distance (m)'); axes[1].set_ylim(0,560); axes[1].set_yticks([0,100,200,300,400,500])
    cmap1=LinearSegmentedColormap.from_list('pastel_accuracy',['#FCF9ED','#F6DDBB','#C1E9DD','#81C8BB'])
    cmap2=LinearSegmentedColormap.from_list('pastel_distance',['#F7F7FC','#D7D7EB','#B3ABD7','#9186BB'])
    heatmap(axes[2],caxes[0],'top1','Top-1 (%)',cmap1,40,100,name,'c')
    heatmap(axes[3],caxes[1],'median_distance_m','Distance (m)',cmap2,100,650,name,'d')
    handles=[Line2D([0],[0],color=darker(c,.72),marker=m,ms=4,mfc=c,mec=darker(c,.65),lw=1.15) for c,m in zip(SCOL,MARK)]
    handles.append(Line2D([0],[0],color=darker(PALETTE['peach'],.72),marker='D',ms=4,mfc=PALETTE['peach'],lw=0))
    f.legend(handles,SENSLABEL+['Deployed reference (30)'],loc='upper center',bbox_to_anchor=(.53,.99),ncol=2,columnspacing=2.5,handlelength=2)
    export(f,name)

partition(); sensor()
pd.DataFrame(data).to_csv(OUTPUT/'source_data/partition_sensor_figure_source_data.csv',index=False)
for name,sub in pd.DataFrame(data).groupby('figure'):
    sub.to_csv(DELIVERY/f'{name}_source_data.csv',index=False)
