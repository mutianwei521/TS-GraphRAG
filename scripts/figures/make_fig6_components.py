"""Fig. 6. Complete component comparisons using reference-sampled pastel fills.

All legacy a/c/d values retained. Panel b uses strict retained-corpus evaluation,
mean and sample SD over five measurement-noise realizations, with no white dots.
"""
import json
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from nature_style import ROOT, SD, QA, figure, export
from pastel_style import apply_style, PHYSICS_COLORS, INK, TEAL, RED, BLUE, GREY, ORANGE, YELLOW

apply_style()
EDGE='#B5B5B5'

eng=json.loads((ROOT/'expH_engine_results.json').read_text(encoding='utf-8'))
phys=json.loads((ROOT/'expH_physics_baselines_results.json').read_text(encoding='utf-8'))
llm=json.loads((ROOT/'expH_llm_results.json').read_text(encoding='utf-8'))
s6=pd.read_csv(SD/'TableS6_paired_tests.csv')
s4=pd.read_csv(SD/'holdout_summary.csv')
legacy_rows=pd.read_csv(SD/'Fig6_legacy_plotted_values.csv')
PROTS=[f'H{i}' for i in range(9)]
N=eng['n_queries']; assert N==phys['n_queries']==llm['n_all']==540
NE,NR=llm['n_errors_total'],llm['recoverable_total']
NSE,NSC=llm['sampled_errors'],llm['sampled_controls']
BASE=100*llm['base_top1']; CEIL=100*(N-NE+NR)/N
assert abs(BASE-100*(N-NE)/N)<1e-9 and (NSE,NSC)==(120,60)
rows=[]
LABEL={'projection':'Projection','cosine 1-NN':'Cosine 1-NN','WLS/NNLS':'WLS / NNLS',
       'Mahalanobis':'Mahalanobis','Bayes-marginal':'Bayes-marginal','KDE':'KDE',
       'graph-reg 1-NN':'Graph variant','label-prop':'Label propagation'}
methods=phys['methods']; assert len(methods)==8
M=np.array([[phys['summary'][h][m] for h in PROTS] for m in methods])
assert M.shape==(8,9) and np.all((M>=0)&(M<=100))
COL={method:PHYSICS_COLORS[method] for method in methods}

fig=figure(170)
ax_a=fig.add_axes([.095,.690,.870,.235])
ax_b=fig.add_axes([.095,.110,.365,.400])
ax_c=fig.add_axes([.620,.405,.335,.145])
ax_d=fig.add_axes([.745,.110,.190,.225])
for label,x,y in [('a',.035,.972),('b',.035,.581),('c',.555,.581),('d',.555,.365)]:
    fig.text(x,y,label,fontsize=8,weight='bold',va='top')
fig.text(.095,.972,'Physics-aware estimators',fontsize=7.5,va='top')
fig.text(.965,.972,'a, c, d: single realization',fontsize=6.5,ha='right',va='top')
fig.text(.095,.581,'Fusion and numerical retrieval',fontsize=7.5,va='top')
fig.text(.095,.552,'Strict pool · 5 noise seeds · mean ± s.d.',fontsize=6.3,va='top')
fig.text(.620,.581,'Graph variant − 1-NN',fontsize=7.5,va='top')
fig.text(.620,.365,'Language-model re-ranking',fontsize=7.5,va='top')

# a: all 8 estimators under all 9 exclusions; triangles identify all tied maxima.
best=[]; width=.88/8
for j,h in enumerate(PROTS):
    top=np.flatnonzero(M[:,j]==M[:,j].max()).tolist()
    best.append([methods[i] for i in top])
    for i,m in enumerate(methods):
        v=float(M[i,j]); x=j+(i-3.5)*width
        ax_a.bar(x,v,width=width*.95,color=COL[m],edgecolor=EDGE,linewidth=.35,
                 hatch='/' if m=='projection' else None,zorder=3)
        if i in top: ax_a.plot(x,v+2.1,marker='v',ms=2.7,color=INK,markeredgewidth=.35,ls='none',zorder=4)
        rows.append(dict(panel='a',protocol=h,method=m,value=v,unit='Top-1 percent',best_in_protocol=i in top))
ax_a.axhline(100/15,color='#8A8A8A',lw=.5,ls=':',zorder=1)
ax_a.set_xlim(-.65,8.65); ax_a.set_ylim(0,103)
ax_a.set_xticks(range(9),PROTS); ax_a.set_yticks([0,20,40,60,80,100])
ax_a.set_ylabel('Top-1 accuracy (%)')
fig.legend(handles=[Patch(fc=COL[m],ec=EDGE,lw=.35,hatch='/' if m=='projection' else None,label=LABEL[m]) for m in methods],
           loc='upper center',bbox_to_anchor=(.54,.652),ncol=4,fontsize=6.5,columnspacing=1.55,
           handlelength=1.65,handletextpad=.5,labelspacing=.35)

# b: 27 bars, including separate projection/cosine bars at equal accuracies.
METHOD_MAP={'engine':'fusion_strict','projection':'projection','cosine':'cosine'}
lookup=s4.set_index(['protocol','method'])
V={key:np.array([lookup.loc[(h,method),'top1_pct_mean'] for h in PROTS]) for key,method in METHOD_MAP.items()}
E={key:np.array([lookup.loc[(h,method),'top1_pct_sd'] for h in PROTS]) for key,method in METHOD_MAP.items()}
assert np.allclose(V['projection'],V['cosine'])
for i,h in enumerate(PROTS):
    ax_b.axhspan(i-.43,i+.43,color=YELLOW,alpha=.23,zorder=0) if i in [2,8] else None
    for k,(key,color) in enumerate(zip(V,[TEAL,COL['projection'],COL['cosine 1-NN']])):
        assert lookup.loc[(h,METHOD_MAP[key]),'n_noise_seeds']==5
        ax_b.barh(i+(k-1)*.25,V[key][i],height=.235,color=color,edgecolor=EDGE,linewidth=.35,
                  hatch='/' if key=='projection' else None,zorder=3,
                  xerr=E[key][i],error_kw={'elinewidth':.45,'capsize':1.2,'capthick':.45,'ecolor':'#666666'})
        rows.append(dict(panel='b',protocol=h,method=key,value=float(V[key][i]),sd=float(E[key][i]),n_noise_seeds=5,unit='Top-1 percent'))
    deficit=V['projection'][i]-V['engine'][i]; assert deficit>0
    ax_b.text(103,i,f'−{deficit:.1f}',ha='left',va='center',fontsize=6.5,clip_on=False)
ax_b.set_yticks(range(9),PROTS); ax_b.set_ylim(8.6,-.6); ax_b.set_xlim(0,100)
ax_b.set_xticks([0,25,50,75,100]); ax_b.set_xlabel('Top-1 accuracy (%)')
ax_b.tick_params(axis='y',length=0,pad=4)
ax_b.text(1.03,1.025,'Deficit\n(pp)',transform=ax_b.transAxes,fontsize=6.3,va='bottom',ha='left')
ax_b.legend(handles=[Patch(fc=TEAL,ec=EDGE,lw=.35,label='Fusion'),Patch(fc=COL['projection'],ec=EDGE,lw=.35,hatch='/',label='Projection'),
                     Patch(fc=COL['cosine 1-NN'],ec=EDGE,lw=.35,label='Cosine 1-NN')],
            loc='lower left',bbox_to_anchor=(-.015,1.012),ncol=3,borderaxespad=0,
            handlelength=1.1,handletextpad=.35,columnspacing=.8,fontsize=6.3)

# c: retain all five published graph contrasts, without causal component language.
g=s6[s6['Comparison (first vs second)'].str.startswith('Graph-regularized')].copy()
assert g.Protocol.tolist()==['H0','H2','H4','H6','H8']
delta=(g['Top-1 first (%)']-g['Top-1 second (%)']).to_numpy(float)
ax_c.axhline(0,color='#666666',lw=.55)
for i,(h,d) in enumerate(zip(g.Protocol,delta)):
    ax_c.bar(i,d,width=.62,color=COL['graph-reg 1-NN'],edgecolor=EDGE,linewidth=.35,zorder=3)
    ax_c.text(i,d-.08,f'{d:+.1f}'.replace('-','−'),ha='center',va='top',fontsize=6.5)
    rows.append(dict(panel='c',protocol=h,method='graph variant minus cosine 1-NN',value=float(d),unit='percentage points'))
ax_c.set_xticks(range(5),g.Protocol); ax_c.set_xlim(-.55,4.55); ax_c.set_ylim(-2.45,.08)
ax_c.set_yticks([-2,-1,0]); ax_c.set_ylabel('Δ Top-1 (pp)',labelpad=2)

# d: raw stratum counts diverge around zero; net remains a weighted estimate.
items=sorted(llm['models'].items(),key=lambda kv:-kv[1]['net_pp']); assert len(items)==7
name_map={'deepseek-v4-flash:cloud':'deepseek-v4-flash','deepseek-v4-pro:cloud':'deepseek-v4-pro',
          'qwen3.5:397b-cloud':'qwen3.5:397b','glm-5.2:cloud':'glm-5.2',
          'gpt-oss:120b-cloud':'gpt-oss:120b','gpt-oss:20b-cloud':'gpt-oss:20b',
          'gemma4:31b':'gemma4:31b','gemma4:31b-cloud':'gemma4:31b','gemma4:31b:cloud':'gemma4:31b'}
llm_rows=[]
ax_d.axvline(0,color='#666666',lw=.55)
for i,(model,v) in enumerate(items):
    assert v['n_errors']==NSE and v['n_controls']==NSC
    rec,brk,net=v['recovered'],v['broken'],v['net_pp']
    recomputed=100*((NE/N)*(rec/NSE)-((N-NE)/N)*(brk/NSC)); assert abs(net-recomputed)<1e-9
    ax_d.barh(i,-brk,height=.62,color=RED,edgecolor=EDGE,linewidth=.35,zorder=2)
    ax_d.barh(i,rec,height=.62,color=TEAL,edgecolor=EDGE,linewidth=.35,zorder=2)
    ax_d.text(-brk-1.3,i,str(brk),ha='right',va='center',fontsize=6)
    ax_d.text(rec+1.3,i,str(rec),ha='left',va='center',fontsize=6)
    ax_d.text(1.205,i,f'{net:+.1f}'.replace('-','−'),transform=ax_d.get_yaxis_transform(),
              va='center',ha='right',fontsize=6.5,clip_on=False)
    llm_rows.append(dict(model=model,corrected_errors=rec,broken_correct=brk,net_pp=net,
                         sampled_errors=NSE,sampled_correct=NSC,population_errors=NE,population_queries=N))
    rows.extend([dict(panel='d',protocol='H2',method=model,value=rec,unit='corrected count'),
                 dict(panel='d',protocol='H2',method=model,value=brk,unit='broken count')])
ax_d.set_yticks(range(7),[name_map.get(k,k.replace(':cloud','')) for k,_ in items])
ax_d.tick_params(axis='y',length=0,pad=4,labelsize=6.3)
ax_d.set_ylim(6.55,-.6); ax_d.set_xlim(-44,29); ax_d.set_xticks([-30,-15,0,15])
ax_d.set_xticklabels(['30','15','0','15']); ax_d.set_xlabel('Answers changed')
ax_d.spines['left'].set_visible(False)
ax_d.text(1.205,1.04,'Net (pp)',fontsize=6.3,transform=ax_d.transAxes,ha='right')
ax_d.legend(handles=[Patch(fc=RED,ec=EDGE,lw=.35,label='Broken'),Patch(fc=TEAL,ec=EDGE,lw=.35,label='Corrected')],
            loc='upper center',bbox_to_anchor=(.38,-.225),ncol=2,fontsize=6.3,
            handlelength=1.2,columnspacing=1,handletextpad=.4)
for ax in [ax_a,ax_b,ax_c,ax_d]: ax.grid(False)

pd.DataFrame(rows).to_csv(SD/'Fig6_all_plotted_values.csv',index=False)
pd.DataFrame(llm_rows).to_csv(SD/'Fig6d_llm_counts_and_weighted_net.csv',index=False)
g.to_csv(SD/'Fig6c_graph_variant_paired_tests.csv',index=False)
assert len(rows)==118
new_legacy=pd.DataFrame(rows).query("panel != 'b'")[['panel','protocol','method','value','unit']].reset_index(drop=True)
old_legacy=legacy_rows.query("panel != 'b'")[['panel','protocol','method','value','unit']].reset_index(drop=True)
pd.testing.assert_frame_equal(new_legacy,old_legacy,check_dtype=False,atol=1e-12)
qa=dict(cells_by_panel={'a':72,'b':27,'c':5,'d_counts':14,'d_net':7},exclusions='none',
        query_count=N,replication='a/c/d original single realization; b mean and sample SD of five independent noise realizations on same 540 fixed queries',
        fusion_table_check='27 exact means/SDs read from strict holdout summary; deficits calculated before rounding',
        unchanged_legacy_value_count=len(new_legacy),white_dots=False,
        llm_net_check='all seven net changes reconstructed from corrections, breakages and population proportions',
        graph_contrast='top-10 rank-weighted voting plus smoothing versus one nearest template; not isolated smoothing',
        best_estimators=dict(zip(PROTS,best)),base_top1=BASE,reranking_ceiling=CEIL,
        interpretation='performance applies to stated configurations; graph differences are not significant after Holm adjustment',
        style='Arial; sampled pastel method fills; 0.35 pt light-grey borders; sparse projection hatch; no numerical heatmap',
        method_colors=COL,bar_border={'color':EDGE,'width_pt':.35})
(QA/'Fig6_data_checks.json').write_text(json.dumps(qa,indent=2),encoding='utf-8')
export(fig,'Fig6_components_holdout')
print(json.dumps(qa))
