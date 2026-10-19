"""Fig. S1: all 300 observations, fixed seed offsets and mean ± sample SD.

Structural adaptation of the existing paired-draw grid. Each actual_seed is a
paired corpus/query-location draw. Horizontal slots never alter source y values.
Shared export writes 183 × 150 mm PDF/SVG and 600 dpi PNG, editable SVG text
(svg.fonttype='none') and embedded PDF fonts (pdf.fonttype=42).
"""
import hashlib
import json
from collections import Counter
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from nature_style import *
from allocation_data import *
from pastel_style import apply_style, darker, BLUE, RED, YELLOW, TEAL, GREY, INK
apply_style()
# Explicitly check the inherited export/style contract without changing it.
required_rc={'font.family':['sans-serif'],'font.sans-serif':['Arial','DejaVu Sans'],
             'font.size':7,'svg.fonttype':'none','pdf.fonttype':42}
assert all(plt.rcParams[key]==value for key,value in required_rc.items())
width_mm=183
dpi=600
assert plt.rcParams['savefig.dpi']==dpi
source_hash=hashlib.sha256((E5/'e5_h0_draws.csv').read_bytes()).hexdigest()
fig=figure(150)
assert abs(fig.get_figwidth()*25.4-width_mm)<1e-10
gs=fig.add_gridspec(2,2,left=.09,right=.98,bottom=.10,top=.93,wspace=.30,hspace=.44)
spec=[('G1','top1','Represented locations','Top-1 accuracy (%)',BLUE),
      ('G2','top1','Unseen locations','Top-1 accuracy (%)',RED),
      ('G2','top3','Unseen locations','Top-3 accuracy (%)',YELLOW),
      ('G2','med_m','Unseen locations','Distance (coordinate units)',TEAL)]
# One fixed slot per seed across designs and panels. The central gap keeps
# the summary stem distinct from observations; there is no vertical jitter.
seeds=sorted(DRAWS.loc[DRAWS.protocol.eq('H0'),'actual_seed'].unique())
assert len(seeds)==15
offsets=np.r_[np.linspace(-.40,-.09,7),np.linspace(.09,.40,8)]
point_rows=[];summary_rows=[];artist_checks=[];trajectory_checks=[]
for g,(group,metric,title,ylab,col),panel in zip(gs,spec,'abcd'):
    ax=fig.add_subplot(g)
    cells=[DRAWS[DRAWS.group.eq(group)&DRAWS.design.eq(d)&DRAWS.protocol.eq('H0')]
           .sort_values('actual_seed') for d in DES]
    assert all(cell.actual_seed.tolist()==seeds for cell in cells)
    matrix=np.array([cell[COL[metric]].to_numpy(float) for cell in cells])
    reference=np.array([values(group,d,metric) for d in DES])
    assert np.array_equal(matrix,reference)
    xvals=np.arange(5)[:,None]+offsets[None,:]
    mean=matrix.mean(axis=1);sd=matrix.std(axis=1,ddof=1)
    for run,seed in enumerate(seeds):
        line,=ax.plot(xvals[:,run],matrix[:,run],color='#A6ADB4',alpha=.42,lw=.35,zorder=1)
        line.set_gid(f'paired_draw_{panel}_seed_{seed}')
        assert np.array_equal(line.get_ydata(),matrix[:,run])
        trajectory_checks.append({'panel':panel,'actual_seed':int(seed),'points':5,'values_exact':True})
    ax.fill_between(range(5),mean-sd,mean+sd,color=col,alpha=.10,zorder=0)
    summary=ax.errorbar(range(5),mean,yerr=sd,fmt='D-',color=darker(col),
                mfc=col,mec=darker(col,.58),mew=.65,ecolor=INK,lw=1.10,
                elinewidth=.65,ms=4.3,capsize=2.7,capthick=.65,zorder=3)
    assert np.array_equal(summary.lines[0].get_ydata(),reference.mean(axis=1))
    assert np.array_equal(sd,reference.std(axis=1,ddof=1))
    for di,segment in enumerate(summary.lines[2][0].get_segments()):
        assert np.array_equal(segment[:,1],[mean[di]-sd[di],mean[di]+sd[di]])
    for di,(design,cell) in enumerate(zip(DES,cells)):
        points=ax.scatter(xvals[di],matrix[di],s=9,facecolors='white',
                          edgecolors='#59636D',linewidths=.55,zorder=5)
        gid=f'draw_points_{panel}_{design}';points.set_gid(gid)
        plotted=np.asarray(points.get_offsets(),dtype=float)
        assert np.array_equal(plotted[:,0],xvals[di])
        assert np.array_equal(plotted[:,1],cell[COL[metric]].to_numpy(float))
        artist_checks.append({'panel':panel,'design':design,'svg_gid':gid,
                              'count':len(plotted),'y_values_exact':True})
        for run,(_,record) in enumerate(cell.iterrows()):
            point_rows.append({'panel':panel,'group':group,'protocol':'H0','design':design,
                'metric':metric,'actual_seed':int(record.actual_seed),'stored_seed':int(record.stored_seed),
                'n_queries':int(record.n_queries),'value':float(record[COL[metric]]),
                'x_offset':float(offsets[run]),'plot_x':float(plotted[run,0]),'plot_y':float(plotted[run,1])})
        stored=SUMMARY[SUMMARY.group.eq(group)&SUMMARY.design.eq(design)&
                       SUMMARY.protocol.eq('H0')&SUMMARY.metric.eq(metric)].iloc[0]
        summary_rows.append({'panel':panel,'group':group,'protocol':'H0','design':design,
            'metric':metric,'n_draws':15,'mean':float(mean[di]),'sample_sd':float(sd[di]),
            'stored_mean':float(stored['mean']),'stored_sample_sd':float(stored.sample_sd),
            'mean_unchanged_from_prior_calculation':True,'sample_sd_unchanged_from_prior_calculation':True})
    ax.set_xticks(range(5),LABELS);ax.set_xlim(-.53,4.53)
    ax.set_ylabel(ylab);ax.set_title(title,loc='left',pad=7)
    if metric=='med_m':ax.set_ylim(0,315);ax.set_yticks([0,100,200,300])
    else:ax.set_ylim(0,105);ax.set_yticks([0,25,50,75,100])
    clean(ax);letter(ax,panel,x=-.18,y=1.14)
fig.legend([plt.Line2D([],[],color='#A6ADB4',lw=.35,marker='o',ms=3,
                       mfc='white',mec='#59636D',mew=.55),
            plt.Line2D([],[],color=darker(TEAL),mfc=TEAL,mec=darker(TEAL,.58),
                       lw=1.1,marker='D',ms=4.3)],
           ['Individual draw (n = 15)','Mean ± s.d.'],loc='lower center',
           bbox_to_anchor=(.53,.015),ncol=2)
pd.DataFrame(point_rows).to_csv(SD/'FigS1_actual_draw_points.csv',index=False,float_format='%.17g')
pd.DataFrame(summary_rows).to_csv(SD/'FigS1_display_summaries.csv',index=False,float_format='%.17g')
roundtrip=pd.read_csv(SD/'FigS1_actual_draw_points.csv',float_precision='round_trip')
assert np.array_equal(roundtrip['value'],[r['value'] for r in point_rows])
assert np.array_equal(roundtrip['plot_y'],roundtrip['value'])
counts=Counter((r['panel'],r['design']) for r in point_rows)
assert len(point_rows)==300 and len(counts)==20 and set(counts.values())=={15}
assert all(r['value']==r['plot_y'] for r in point_rows)
assert all(len({r['plot_x'] for r in point_rows if (r['panel'],r['design'])==cell})==15 for cell in counts)
export(fig,'FigS1_draw_level')
assert (FIGDIR/'FigS1_draw_level.pdf').read_bytes().startswith(b'%PDF')
with Image.open(FIGDIR/'FigS1_draw_level.png') as raster:
    raster_size=list(raster.size);raster_dpi=list(raster.info['dpi'])
    assert all(abs(value-dpi)<.02 for value in raster_dpi)
svg=ET.parse(FIGDIR/'FigS1_draw_level.svg');ns={'svg':'http://www.w3.org/2000/svg'}
for check in artist_checks:
    group_element=next(el for el in svg.iter() if el.get('id')==check['svg_gid'])
    check['svg_marker_count']=len(group_element.findall('.//svg:use',ns))
    assert check['svg_marker_count']==15
text_qa=json.loads((QA/'FigS1_draw_level.json').read_text(encoding='utf-8'))
assert not text_qa['overflow']
report={'figure':'FigS1_draw_level','observation_count':300,'cell_count':20,'draws_per_cell':15,
    'actual_seeds':[int(s) for s in seeds],'all_plot_y_equal_source':True,'vertical_jitter':False,
    'horizontal_offsets':[float(v) for v in offsets],
    'offset_rule':'Fixed slot per actual_seed across every design and panel, with a central summary gap.',
    'trajectory_count':len(trajectory_checks),'trajectory_checks':trajectory_checks,
    'artist_checks':artist_checks,'svg_observation_marker_count':300,'summary_checks':summary_rows,
    'mean_and_sample_sd_unchanged':True,'sample_sd_ddof':1,'excluded_observations':0,
    'source_sha256':source_hash,
    'source_unchanged':source_hash==hashlib.sha256((E5/'e5_h0_draws.csv').read_bytes()).hexdigest(),
    'text_overflow':text_qa['overflow'],'width_mm':text_qa['width_mm'],
    'height_mm':text_qa['height_mm'],'min_text_pt':text_qa['min_text_pt']}
report['raster_pixels']=raster_size;report['raster_dpi']=raster_dpi
report['exported_point_csv_roundtrip_exact']=True
report['max_stored_summary_mean_difference']=max(abs(r['mean']-r['stored_mean']) for r in summary_rows)
report['max_stored_summary_sd_difference']=max(abs(r['sample_sd']-r['stored_sample_sd']) for r in summary_rows)
assert report['source_unchanged']
(QA/'FigS1_observation_points_checks.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
print('FigS1 points: 300; cells: 20 × 15; source y exact; mean/SD unchanged; no exclusions.')
