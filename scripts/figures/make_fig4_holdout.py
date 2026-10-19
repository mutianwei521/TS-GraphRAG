"""Figure 4: reference-sampled pastel fills and fine sample-SD caps.

Structural adaptation of the preceding complete-data figure. The frozen source
contains all 138 summaries and 690 run values; nothing is sampled or excluded.
The graphical grammar is inherited from the user's older quantitative figures.
"""
import json
import hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from nature_style import ROOT, PKG, SD, QA, figure, export
from pastel_style import apply_style, METHOD_COLORS, INK, RED, BLUE, TEAL, ORANGE, YELLOW, GREY

apply_style()
EDGE='#B5B5B5'

DATA=pd.read_csv(SD/'Fig4_all_plotted_cells_and_runs.csv')
assert len(DATA)==690
SEEDS=[41,581,1121,1661,2201]
E4=ROOT/'results/E4_mean_std_20260909'
raw=pd.read_csv(E4/'all_run_accuracies.csv')
ta=pd.read_csv(E4/'Table_S3a_matched90.tsv',sep='\t',encoding='utf-8-sig')
tb=pd.read_csv(E4/'Table_S3b_all540.tsv',sep='\t',encoding='utf-8-sig')
MAP={'Projection residual':'projection','Cosine 1-NN':'cosine 1-NN','Cosine k-NN (k=5)':'k-NN (k=5)',
     'Random Forest (100 trees)':'Random Forest','SVM (RBF)':'SVM (RBF)','MLP (128,64)':'MLP',
     'Nearest Centroid':'Nearest Centroid','Logistic Regression':'Logistic Regression',
     'GCN+':'GCN+','GAT+':'GAT+','GraphSAGE+':'GraphSAGE+'}
SHORT={'Projection residual':'Projection','Cosine 1-NN':'Cosine 1-NN','Cosine k-NN (k=5)':'Cosine 5-NN',
       'Random Forest (100 trees)':'Random forest','SVM (RBF)':'SVM','MLP (128,64)':'MLP',
       'Nearest Centroid':'Centroid','Logistic Regression':'Logistic','GCN+':'GCN+','GAT+':'GAT+','GraphSAGE+':'GraphSAGE+'}
COL={method:METHOD_COLORS[method] for method in SHORT}
PROTS=[f'H{i}' for i in range(9)]
CHECKS=[]
POINT_RECORDS=[]
POINT_ARTISTS=[]
SUMMARY_RECORDS=[]

def cell(panel,h,m):
    s=DATA[(DATA.panel==panel)&(DATA.protocol==h)&(DATA.method==m)].sort_values('run_seed')
    assert s.run_seed.tolist()==SEEDS,(panel,h,m)
    vals=s.accuracy_percent.to_numpy(float)
    mean=float(s.mean_percent.iloc[0]); sd=float(s.sample_sd_percent.iloc[0])
    assert s.mean_percent.nunique()==s.sample_sd_percent.nunique()==1
    dm,ds=abs(vals.mean()-mean),abs(vals.std(ddof=1)-sd)
    assert dm<=.06 and ds<=.06
    if panel!='d':
        tab=tb if panel=='a' else ta
        mt,st=map(float,tab.loc[tab.Method==m,h].iloc[0].split('±'))
        assert abs(mt-mean)<1e-9 and abs(st-sd)<1e-9
        rp='classical540' if panel=='a' else 'matched90'
        r=raw[(raw.panel==rp)&(raw.protocol==h)&(raw.method==MAP[m])].sort_values('run_seed')
        assert np.allclose(vals,r.accuracy_percent,rtol=0,atol=1e-10)
    CHECKS.append((panel,h,m,float(dm),float(ds)))
    return mean,sd,vals

def packed_offsets(value_pt,width_pt):
    """Fit five circles inside a bar lane without perturbing observed values.

    Values are in screen points only for collision geometry. All plotted
    numerical coordinates remain the unmodified observations. Reduce glyph
    diameter only where a narrow lane and overlapping values require it.
    """
    order=np.argsort(value_pt,kind='stable')
    gap=.12
    for diameter in np.linspace(2.0,width_pt/6,42):
        bound=(width_pt-diameter)/2-.035
        if bound<=0:
            continue
        placed={}
        def place(k):
            if k==len(order):
                return True
            idx=int(order[k]); candidates=[0.,-bound,bound]
            for other,x in placed.items():
                dy=abs(value_pt[idx]-value_pt[other])
                if dy<diameter+gap:
                    dx=np.sqrt((diameter+gap)**2-dy**2)+1e-7
                    candidates.extend([x-dx,x+dx])
            candidates=sorted(set(candidates),key=lambda x:(abs(x),x))
            for x in candidates:
                if abs(x)>bound+1e-8:
                    continue
                if any(np.hypot(x-p,value_pt[idx]-value_pt[other])<diameter+gap-1e-8
                       for other,p in placed.items()):
                    continue
                placed[idx]=x
                if place(k+1):
                    return True
                del placed[idx]
            return False
        if place(0):
            return np.array([placed[k] for k in range(len(value_pt))]),float(diameter)
    raise AssertionError('Five observations could not be packed in their bar lane')


def draw_runs(ax,pos,vals,width,tag,*,horizontal=False):
    """Deterministic categorical packing; no jitter on the measured-value axis."""
    span=np.diff(ax.get_ylim() if horizontal else ax.get_xlim())[0]
    pixels=ax.bbox.height if horizontal else ax.bbox.width
    width_pt=abs(width/span)*pixels*72/fig.dpi
    stroke_pt=.22
    value_xy=np.column_stack([vals,np.full(5,pos)]) if horizontal else np.column_stack([np.full(5,pos),vals])
    value_pt=ax.transData.transform(value_xy)[:,0 if horizontal else 1]*72/fig.dpi
    offsets_pt,diameter_pt=packed_offsets(value_pt,width_pt)
    marker_pt=diameter_pt-stroke_pt
    assert marker_pt>.35,(tag,width_pt)
    offsets=offsets_pt/width_pt*width
    xs,ys=(vals,pos+offsets) if horizontal else (pos+offsets,vals)
    artist=ax.scatter(xs,ys,s=marker_pt**2,marker='o',facecolors='white',
                      edgecolors=INK,linewidths=stroke_pt,zorder=6)
    artist.set_gid('runs_'+ '_'.join(tag).replace(' ','_'))
    actual=np.asarray(artist.get_offsets(),float)
    assert np.array_equal(actual[:,0 if horizontal else 1],vals)
    disp=ax.transData.transform(actual)*72/fig.dpi
    pair_dist=np.linalg.norm(disp[:,None,:]-disp[None,:,:],axis=-1)
    np.fill_diagonal(pair_dist,np.inf)
    assert pair_dist.min()>diameter_pt,(tag,pair_dist.min(),diameter_pt)
    assert abs(offsets_pt).max()+diameter_pt/2 < width_pt/2
    POINT_ARTISTS.append((tag,artist))
    panel,protocol,method=tag
    for k,seed in enumerate(SEEDS):
        POINT_RECORDS.append(dict(panel=panel,protocol=protocol,method=method,
            run_seed=seed,accuracy_percent=float(vals[k]),plot_x=float(actual[k,0]),
            plot_y=float(actual[k,1]),value_axis='x' if horizontal else 'y',
            category_center=float(pos),category_offset=float(offsets[k]),
            marker_core_pt=float(marker_pt),marker_stroke_pt=stroke_pt,
            marker_outer_diameter_pt=float(diameter_pt),bar_width_pt=float(width_pt),
            minimum_same_cell_marker_gap_pt=float(pair_dist.min()-diameter_pt)))


def bar(ax,pos,mean,sd,vals,color,width,*,horizontal=False,points=False,hatch=None,pale=False,tag=None):
    fc=plt.matplotlib.colors.to_rgba(color,.42) if pale else color
    kw=dict(color=fc,edgecolor=EDGE,linewidth=.35,hatch=hatch,zorder=2)
    ek=dict(ecolor='#626262',elinewidth=.55,capsize=1.3,capthick=.5,zorder=4)
    if horizontal:
        bars=ax.barh(pos,mean,height=width,**kw)
        errors=ax.errorbar(mean,pos,xerr=sd,fmt='none',**ek)
        if points: draw_runs(ax,pos,vals,width,tag,horizontal=True)
    else:
        bars=ax.bar(pos,mean,width=width,**kw)
        errors=ax.errorbar(pos,mean,yerr=sd,fmt='none',**ek)
        if points: draw_runs(ax,pos,vals,width,tag)
    drawn_mean=bars.patches[0].get_width() if horizontal else bars.patches[0].get_height()
    segment=errors.lines[2][0].get_segments()[0]
    bounds=segment[:,0 if horizontal else 1]
    assert drawn_mean==mean
    assert np.allclose(bounds,[mean-sd,mean+sd],rtol=0,atol=1e-12)
    SUMMARY_RECORDS.append(dict(panel=tag[0],protocol=tag[1],method=tag[2],
        mean_percent=mean,sample_sd_percent=sd,drawn_mean_percent=float(drawn_mean),
        drawn_error_low=float(bounds[0]),drawn_error_high=float(bounds[1])))

fig=figure(170)
a=fig.add_axes([.095,.680,.870,.250])
b=fig.add_axes([.170,.205,.205,.325])
c=fig.add_axes([.430,.205,.375,.325])
d=fig.add_axes([.865,.205,.105,.325])
# Fix scales before deriving marker diameters in physical points.
a.set_xlim(-.65,8.65);a.set_ylim(0,103)
b.set_xlim(0,105);b.set_ylim(10.65,-.65)
c.set_xlim(-.65,4.65);c.set_ylim(0,103)
d.set_xlim(-.6,2.6);d.set_ylim(0,103)
for label,x,y in [('a',.035,.973),('b',.035,.570),('c',.390,.570),('d',.820,.570)]:
    fig.text(x,y,label,weight='bold',fontsize=8,va='top')
for x,y,title in [(.095,.971,'Reference queries'),(.17,.567,'H0 versus H2'),(.43,.567,'Other holdouts'),(.865,.567,'Destinations')]:
    fig.text(x,y,title,fontsize=7.5,va='top')
fig.text(.965,.971,'540 queries · 5 runs',fontsize=6.5,ha='right',va='top')

# a: every one of the 8 x 9 original bars, with mean and sample SD.
ma=DATA[DATA.panel=='a'].method.drop_duplicates().tolist();assert len(ma)==8
w=.88/len(ma)
for j,h in enumerate(PROTS):
    for i,m in enumerate(ma):
        mean,sd,vals=cell('a',h,m)
        bar(a,j-.44+(i+.5)*w,mean,sd,vals,COL[m],w,points=True,
            hatch='/' if m=='Projection residual' else None,tag=('a',h,m))
a.axhline(100/15,ls=':',color='#8A8A8A',lw=.5,zorder=1)
a.set_xlim(-.65,8.65);a.set_ylim(0,103);a.set_yticks([0,20,40,60,80,100]);a.set_xticks(range(9),PROTS)
a.set_ylabel('Top-1 accuracy (%)');a.tick_params(pad=2)
fig.legend(handles=[Patch(fc=COL[m],ec=EDGE,lw=.35,hatch='/' if m=='Projection residual' else None,label=SHORT[m]) for m in SHORT],
           loc='upper center',bbox_to_anchor=(.535,.645),ncol=4,columnspacing=1.7,handlelength=1.7,handletextpad=.55,fontsize=6.4)

# b: every method, paired bars under H0 and H2, including all five run points.
mb=DATA[DATA.panel=='b'].method.drop_duplicates().tolist()
bd={m:{h:cell('b',h,m) for h in ['H0','H2']} for m in mb}
mb=sorted(mb,key=lambda m:(-bd[m]['H2'][0],m!='Projection residual'))
b.axhspan(-.46,.46,color='#F6F3FB',zorder=0)
for i,m in enumerate(mb):
    for h,off,pale in [('H0',-.185,True),('H2',.185,False)]:
        mean,sd,vals=bd[m][h]
        bar(b,i+off,mean,sd,vals,COL[m],.33,horizontal=True,points=True,pale=pale,
            hatch='/' if m=='Projection residual' else None,tag=('b',h,m))
b.set_xlim(0,105);b.set_ylim(10.65,-.65);b.set_yticks(range(11),[SHORT[m] for m in mb]);b.tick_params(axis='y',length=0,pad=3)
b.set_xticks([0,25,50,75,100]);b.set_xlabel('Top-1 (%)')
b.legend(handles=[Patch(fc='#ECE8F5',ec=EDGE,lw=.35,label='H0'),Patch(fc=METHOD_COLORS['Projection residual'],ec=EDGE,lw=.35,label='H2')],
         loc='upper center',bbox_to_anchor=(.5,-.18),ncol=2,handlelength=1.3,columnspacing=1.2,fontsize=6.4)

# c: the same complete seven-method, five-protocol comparison.
mc=['Projection residual','Cosine k-NN (k=5)','Random Forest (100 trees)',
    'Logistic Regression','GraphSAGE+','GAT+','GCN+'];hc=['H0','H2','H3','H4','H8'];wc=.88/7
for j,h in enumerate(hc):
    for i,m in enumerate(mc):
        mean,sd,vals=cell('c',h,m)
        bar(c,j-.44+(i+.5)*wc,mean,sd,vals,COL[m],wc,points=True,
            hatch='/' if m=='Projection residual' else None,tag=('c',h,m))
c.set_xlim(-.65,4.65);c.set_ylim(0,103);c.set_xticks(range(5),hc);c.set_yticks([0,20,40,60,80,100]);c.set_ylabel('Top-1 (%)')
fig.text(.43,.143,'90 fixed queries',ha='left',fontsize=6.4)

# d: all nine destination shares, mean/SD and all five repetition points.
cats=[('Correct','correct_pct',BLUE),('Adjacent','adjacent_error_pct',ORANGE),('Other','nonadjacent_error_pct',RED)]
for j,h in enumerate(['H0','H2','H8']):
    for i,(label,m,col) in enumerate(cats):
        mean,sd,vals=cell('d',h,m)
        bar(d,j+(i-1)*.26,mean,sd,vals,col,.24,points=True,tag=('d',h,m))
d.set_xlim(-.6,2.6);d.set_ylim(0,103);d.set_xticks(range(3),['H0','H2','H8']);d.set_yticks([0,50,100]);d.set_xlabel('Share (%)')
d.legend(handles=[Patch(fc=col,ec=EDGE,lw=.35,label=label) for label,_,col in cats],
         loc='upper center',bbox_to_anchor=(.5,-.18),ncol=1,handlelength=1.3,fontsize=6.2)
for ax in [a,b,c,d]:
    ax.spines[['top','right']].set_visible(False);ax.tick_params(labelsize=6.5);ax.set_axisbelow(True)
    ax.grid(False)
fig.text(.095,.070,'Open circles: five individual runs per bar   |   Bars and caps: mean ± sample SD',
         fontsize=6.5,ha='left')
assert len(CHECKS)==138
plotted=pd.DataFrame(POINT_RECORDS)
keys=['panel','protocol','method','run_seed']
assert len(plotted)==sum(len(p.get_offsets()) for _,p in POINT_ARTISTS)==690
assert plotted.groupby(keys[:-1]).size().eq(5).all()
assert not plotted.duplicated(keys).any()
matched=DATA.merge(plotted,on=keys,validate='one_to_one',suffixes=('_source','_plotted'))
assert np.array_equal(matched.accuracy_percent_source,matched.accuracy_percent_plotted)
baseline=ROOT/'plot_tables/Fig4_all_plotted_cells_and_runs.csv'
source_hash=hashlib.sha256((SD/'Fig4_all_plotted_cells_and_runs.csv').read_bytes()).hexdigest()
assert source_hash==hashlib.sha256(baseline.read_bytes()).hexdigest()
plotted.to_csv(QA/'Fig4_observation_points_coordinates.csv',index=False)
pd.DataFrame(SUMMARY_RECORDS).to_csv(QA/'Fig4_observation_summary_coordinates.csv',index=False)
qa=dict(cells_by_panel={p:sum(x[0]==p for x in CHECKS) for p in ['a','b','c','d']},
        source_run_values=len(DATA),mean_max_rounding_difference=max(x[3] for x in CHECKS),
        sd_max_rounding_difference=max(x[4] for x in CHECKS),excluded_observations=0,
        SD='sample SD of five fixed-query runs',run_points='all five plotted in every cell of a/b/c/d',
        rendered_point_counts=plotted.groupby('panel').size().to_dict(),
        source_sha256=source_hash,source_byte_identical_to_preceding=True,
        all_138_drawn_means_and_SD_bounds_match_source=True,
        point_value_coordinates_exact=True,point_numeric_axis_jitter=0,
        categorical_offsets='deterministic circle packing inside each bar lane; along x in a/c/d and y in b; no value-axis jitter',
        same_cell_marker_overlap=False,markers_stay_within_own_bar_lane=True,
        minimum_marker_gap_pt=float(plotted.minimum_same_cell_marker_gap_pt.min()),
        marker_outer_diameter_pt_range=[float(plotted.marker_outer_diameter_pt.min()),float(plotted.marker_outer_diameter_pt.max())],
        previous_c_width_mm=.240*183,current_c_width_mm=.375*183,
        style='Arial; sampled pastel method fills; 0.35 pt light-grey borders; fine sample-SD caps; sparse projection hatch; no numerical heatmap',
        method_colors=COL,bar_border={'color':EDGE,'width_pt':.35},error_caps={'width_pt':.5,'line_width_pt':.55})
(QA/'Fig4_data_checks.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
export(fig,'Fig4_holdout_evaluation');print(json.dumps(qa))
