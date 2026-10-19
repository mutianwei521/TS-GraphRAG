"""Compact Fig. 2 with native vector methods and actual Blender icon renders.

Network geometry, pressure curves and retrieval example are schematic.
Panel c includes all 540 x 30 signed simulated fingerprints; display normalization only.
Device/resource pictograms reuse rendered RGBA assets. The corpus matrix is
losslessly rasterized; relationships, row borders, mathematics and labels are vector.
"""
from pathlib import Path
import json
import re
import math

import numpy as np
from scipy.interpolate import PchipInterpolator
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyBboxPatch, Ellipse, Polygon, FancyArrowPatch
from matplotlib.path import Path as MplPath
from pypdf import PdfReader, PdfWriter
from pypdf.generic import BooleanObject, NameObject

from _paths import prepare_fig2
ROOT = prepare_fig2()
for part in ['figures','notes','qa','source_data']:
    (ROOT/part).mkdir(exist_ok=True,parents=True)
width_mm=183
height_mm=221
W,H=width_mm,height_mm
INK='#26313B'; MUTED='#64717D'; BLUE='#527BAD'; EDGE='#3A4650'
GREEN='#419464'; TEAL='#169AA0'; GOLD='#DEA92B'; DBLUE='#668ECD'
P1='#F0CF58'; P2='#E98780'; P3='#70AFE0'; PURPLE='#8E79C1'
PART=[P1,P2,P3]
plt.rcParams.update({'font.family':'Arial','font.size':6.5,
                     'mathtext.fontset':'stixsans','pdf.fonttype':42,
                     'ps.fonttype':42,'svg.fonttype':'none',
                     'image.composite_image':False,'axes.unicode_minus':False,'savefig.facecolor':'white'})
fig=plt.figure(figsize=(W/25.4,H/25.4),dpi=300)
ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,W),ylim=(H,0));ax.axis('off')
elements=[];texts=[];images=[];panel='';YSHIFT=0

def add(e):
    e['id']=f'{panel}_{len(elements):04d}'
    elements.append(e);return len(elements)+1

def image_asset(name,x,y,w,h):
    y+=YSHIFT
    path=ROOT/'assets'/f'{name}.png'
    img=plt.imread(path)
    assert img.shape[2]==4, f'Expected RGBA: {path}'
    assert abs((img.shape[1]/img.shape[0])/(w/h)-1)<.015, 'Do not stretch 3D renders'
    z=add(dict(type='image',path=path.relative_to(ROOT).as_posix(),x=x,y=y,w=w,h=h,
               source='Blender Cycles',scientific_status='schematic icon'))
    # Composite to its uniform card background before downsampling to avoid alpha fringes.
    bg={'a':'#F0F7F0','b':'#EEF8F8','c':'#FFFAEC'}.get(panel,'#FFFFFF')
    if name=='document': bg='#F7F6F9'
    rgba=img;alpha=rgba[:,:,3:4]
    rendered=np.clip(rgba[:,:,:3]*alpha+np.array(matplotlib.colors.to_rgb(bg))*(1-alpha),0,1)
    elements[-1]['display_compositing']='Original RGBA over '+bg+' before Lanczos resampling'
    artist=ax.imshow(rendered,extent=(x,x+w,y+h,y),origin='upper',aspect='auto',
                     interpolation='lanczos',resample=True,zorder=z)
    images.append((artist,img))

def text(x,y,s,size=6.5,color=INK,weight='normal',ha='left',va='center',**extras):
    y+=YSHIFT
    size=max(6.5,size)
    z=add(dict(type='text',x=x,y=y,text=s,size=size,color=color,weight=weight,ha=ha,va=va,**extras))
    a=ax.text(x,y,s,fontsize=size,color=color,weight=weight,ha=ha,va=va,zorder=z,linespacing=1.05)
    texts.append((elements[-1],a))

def rect(x,y,w,h,fc='none',ec='none',lw=.6,radius=0,**extras):
    y+=YSHIFT
    z=add(dict(type='rect',x=x,y=y,w=w,h=h,fc=fc,ec=ec,lw=lw,radius=radius,**extras))
    p=FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={radius}',
                    fc=fc,ec=ec,lw=lw,zorder=z) if radius else Rectangle((x,y),w,h,fc=fc,ec=ec,lw=lw,zorder=z)
    ax.add_patch(p)

def ellipse(x,y,w,h,fc='white',ec=EDGE,lw=.7,**extras):
    y+=YSHIFT
    z=add(dict(type='ellipse',x=x,y=y,w=w,h=h,fc=fc,ec=ec,lw=lw,**extras))
    ax.add_patch(Ellipse((x+w/2,y+h/2),w,h,fc=fc,ec=ec,lw=lw,zorder=z))

def circle(x,y,r,fc='white',ec=EDGE,lw=.7):
    ellipse(x-r,y-r,2*r,2*r,fc,ec,lw)

def polygon(points,fc='white',ec=EDGE,lw=.7,**extras):
    points=[[xx,yy+YSHIFT] for xx,yy in points]
    z=add(dict(type='polygon',points=points,fc=fc,ec=ec,lw=lw,**extras))
    ax.add_patch(Polygon(points,closed=True,fc=fc,ec=ec,lw=lw,zorder=z))

def line(points,color=EDGE,lw=.7,dashed=False,arrow_end=False,**extras):
    points=[[xx,yy+YSHIFT] for xx,yy in points]
    z=add(dict(type='line' if len(points)==2 else 'polyline',points=points,color=color,
               lw=lw,dashed=dashed,arrow_end=arrow_end,**extras))
    if arrow_end:
        p=MplPath(points,[MplPath.MOVETO]+[MplPath.LINETO]*(len(points)-1))
        ax.add_patch(FancyArrowPatch(path=p,arrowstyle='-|>',mutation_scale=6.5,
                                    lw=lw,color=color,linestyle=(0,(2.4,1.7)) if dashed else '-',zorder=z))
    else:
        p=np.asarray(points)
        ax.plot(p[:,0],p[:,1],color=color,lw=lw,linestyle=(0,(2.4,1.7)) if dashed else '-',
                solid_capstyle='round',zorder=z)

def card(x,y,w,h,title,color,size=6.7,ink='white'):
    rect(x,y,w,h,fc='white',ec=color,lw=.75,radius=2)
    rect(x,y,w,6.7,fc=color,radius=2)
    rect(x,y+3.2,w,3.5,fc=color)
    text(x+w/2,y+3.35,title,size=size,color=ink,weight='bold',ha='center')

def star(x,y,r=1.6,color=GOLD):
    points=[]
    for j in range(10):
        th=-math.pi/2+j*math.pi/5; rr=r if j%2==0 else r*.43
        points.append([x+rr*math.cos(th),y+rr*math.sin(th)])
    polygon(points,fc=color,ec=color,lw=.35)

def strip(x,y,w,h,n=12,family='blue',phase=0):
    # Discrete schematic marks: shade changes have no numeric colour scale.
    palettes={'blue':['#DCEAF5','#AFD0EA','#75ADD6','#BFDCEF','#8CBCE0'],
              'orange':['#F8D7AA','#EFB775','#F6C78F','#E7A963'],
              'green':['#D5E7B4','#B4D27C','#91BB4B'],
              'coral':['#F2C5C2','#EAA3A0','#E18D89']}
    col=palettes[family]
    for k in range(n):
        rect(x+k*w/n,y,w/n,h,fc=col[(k+phase)%len(col)],ec=EDGE,lw=.25)

def sensor(x,y,r=1.55):
    circle(x,y,r,'white','#267A86',.65)
    pts=[[-.85,0],[-.45,0],[-.18,-.63],[.16,.59],[.42,-.18],[.83,-.18]]
    line([[x+a*r*.9,y+b*r*.9] for a,b in pts],color='#267A86',lw=.65)

NODES=np.array([[.10,.20],[.43,.06],[.34,.39],[.07,.59],[.34,.73],[.13,.96],
                [.64,.52],[.88,.24],[.92,.86], [.27,.15],[.04,.38],
                [.51,.92],[.68,.15],[.79,.65],[.53,.32]])
EDGES=[(0,2),(1,2),(2,3),(0,3),(2,4),(4,5),(2,6),(6,7),(6,8),(7,8),
       (0,9),(9,1),(9,2),(0,10),(10,3),(4,11),(11,6),(1,12),(12,7),(12,14),(14,6),(2,14),(6,13),(13,8)]
GROUP=[0,0,0,0,1,1,2,2,2,0,0,1,2,2,0]

def network(x,y,w,h,partitioned=True,sensors=False,labels=True,r=1.6,highlight=None):
    xy=NODES*np.array([w,h])+np.array([x,y])
    for u,v in EDGES:
        line([xy[u].tolist(),xy[v].tolist()],color='#56616B',lw=.7)
    for i,(xx,yy) in enumerate(xy):
        fill=PART[GROUP[i]] if partitioned else PURPLE
        if highlight is not None and GROUP[i]!=highlight: fill='#E8EDF1'
        circle(float(xx),float(yy),r,fill,EDGE,.65)
        if sensors and i in (2,4,6):sensor(float(xx),float(yy),r*.72)
    if labels:
        text(x+w*.0,y-h*.03,'P1',size=6.2,weight='bold')
        text(x+w*.0,y+h*1.09,'P2',size=6.2,weight='bold')
        text(x+w*.95,y+h*.61,'P3',size=6.2,weight='bold')
    return xy

def monitor(x,y,s=5):
    rect(x,y,s*1.1,s*.75,fc='#E5F2F6',ec=EDGE,lw=.65,radius=.3)
    for p,q in [((.18,.5),(.45,.22)),((.45,.22),(.78,.48)),((.18,.5),(.78,.48))]:
        line([[x+s*p[0],y+s*p[1]],[x+s*q[0],y+s*q[1]]],color=TEAL,lw=.5)
    for px,py in [(.18,.5),(.45,.22),(.78,.48)]:circle(x+s*px,y+s*py,s*.065,TEAL,TEAL,.3)
    line([[x+s*.55,y+s*.78],[x+s*.55,y+s*.98]],lw=.8)
    line([[x+s*.28,y+s],[x+s*.82,y+s]],lw=.8)

def document(x,y,w=4,h=5):
    rect(x,y,w,h,fc='#FFF9E9',ec=EDGE,lw=.55,radius=.2)
    polygon([[x+w-1.2,y],[x+w-1.2,y+1.2],[x+w,y+1.2]],fc='#F0D8AC',ec=EDGE,lw=.4)
    for j in range(3):line([[x+.6,y+1.9+j*.85],[x+w-.55,y+1.9+j*.85]],color='#83919A',lw=.55)

def droplet(x,y,s=3,color=P3):
    points=[[x,y-s*.63],[x+s*.40,y-s*.05],[x+s*.43,y+s*.24],
            [x+s*.25,y+s*.5],[x,y+s*.58],[x-s*.25,y+s*.5],
            [x-s*.43,y+s*.24],[x-s*.4,y-s*.05]]
    polygon(points,fc=color,ec=EDGE,lw=.45)

def pin(x,y,s=3):
    polygon([[x-s*.34,y],[x+s*.34,y],[x,y+s*.72]],fc=P2,ec=P2,lw=.4)
    circle(x,y,s*.4,P2,P2,.4);circle(x,y,s*.15,'white','white',.2)

def pump(x,y,s=5):
    line([[x-s*.9,y],[x+s*.9,y]],color='#4C798F',lw=1.1)
    circle(x,y,s*.4,'#DDECF4',EDGE,.6)
    polygon([[x-s*.18,y-s*.21],[x+s*.25,y],[x-s*.18,y+s*.21]],fc='#6897B6',ec='#6897B6',lw=.3)
    line([[x-s*.35,y+s*.48],[x+s*.35,y+s*.48]],lw=.6)

def tank(x,y,s=5):
    rect(x-s*.45,y-s*.46,s*.9,s*.73,fc='#E1EFF6',ec=EDGE,lw=.6)
    line([[x-s*.4,y-s*.05],[x+s*.4,y-s*.05]],color='#6BA8C7',lw=.8)
    for q in [-.32,.32]:line([[x+s*q,y+s*.28],[x+s*q,y+s*.7]],lw=.6)
    line([[x-s*.48,y+s*.7],[x+s*.48,y+s*.7]],lw=.6)

def wave(x,y,w=4,h=2,color=BLUE):
    t=np.linspace(0,1,21)
    line([[x+u*w,y+math.sin(u*2*math.pi)*h/2] for u in t],color=color,lw=.8)

def heading(x,y,letter,title,icon='monitor'):
    text(x,y,letter,size=11.5,weight='bold',color='#10171C')
    if icon=='monitor': monitor(x+7,y-2.5,4.8)
    elif icon=='layers':
        image_asset('knowledge_layers',x+6.55,y-3.3,7.15,6.5)
    elif icon=='doc':document(x+7,y-2.7,4.5,5.4)
    elif icon=='vector':
        for j in range(3):strip(x+7,y-2.5+j*1.6,5,1.2,4,'blue',j)
    text(x+15,y,title,size=9.3,color=BLUE,weight='bold')

def step_title(x,y,s):
    text(x,y,s,size=6.5,weight='bold',ha='center')

def arr(x0,y0,x1,y1,color=TEAL,lw=.9):
    line([[x0,y0],[x1,y1]],color=color,lw=lw,arrow_end=True)

def micro_graph(x,y,kind):
    if kind=='move':
        ellipse(x-1.4,y-1.3,6.8,7.0,fc='#FAEDB9',ec='none')
        ellipse(x+6.6,y-2.0,7.0,9.0,fc='#DFEDF9',ec='none')
        pts=[(0,0),(4,0),(2,4),(8,2),(11,-1),(12,5)]
        edges=[(0,1),(0,2),(1,2),(2,3),(3,4),(3,5),(4,5)]
        for i,j in edges:line([[x+pts[i][0],y+pts[i][1]],[x+pts[j][0],y+pts[j][1]]],lw=.45)
        for i,(xx,yy) in enumerate(pts):circle(x+xx,y+yy,.9,P1 if i<3 else P3,EDGE,.45)
        circle(x+2,y+4,1.35,'none',PURPLE,.75)
        arr(x+3.5,y+5.4,x+7.5,y+3.1,PURPLE,.75)
    elif kind=='refine':
        ellipse(x-1.5,y-1.5,6,9,fc='#FAEDB9',ec='none')
        ellipse(x+6.5,y-1.5,6,9,fc='#DFEDF9',ec='none')
        pts=[(0,0),(3,3),(0,6),(8,0),(11,3),(8,6)]
        for i,j in [(0,1),(1,2),(0,2),(3,4),(4,5),(3,5)]:line([[x+pts[i][0],y+pts[i][1]],[x+pts[j][0],y+pts[j][1]]],lw=.45)
        for i,(xx,yy) in enumerate(pts):circle(x+xx,y+yy,.9,P1 if i<3 else P3,EDGE,.45)
        line([[x+5.4,y-1],[x+5.4,y+7]],color=GREEN,lw=.55,dashed=True)
    else:
        ellipse(x-2,y-1.1,6.2,6.2,fc='#FAEDB9',ec='none')
        ellipse(x+6.9,y-1.1,6.2,6.2,fc='#DFEDF9',ec='none')
        line([[x+1,y+2],[x+10,y+2]],color=EDGE,lw=1.35)
        circle(x+1,y+2,2.2,P1,EDGE,.6);circle(x+10,y+2,2.2,P3,EDGE,.6)
        for cx in [x+1,x+10]:
            for dx,dy in [(-.75,.45),(.75,.45),(0,-.7)]:
                circle(cx+dx,y+2+dy,.38,'white','none',.1)

# Diagram coordinates are in millimetres with a top-left origin.
from matplotlib.colors import LinearSegmentedColormap, Normalize
from PIL import Image
import hashlib

STAGE=[GREEN,TEAL,GOLD,PURPLE]
TINT=['#F0F7F0','#EEF8F8','#FFFAEC','#F4F1FA']
def stage(letter,y,title,color,tag):
    circle(8.5,y,3.5,color,color,.3)
    text(8.5,y,letter,11,color='white',weight='bold',ha='center')
    text(15,y,title,9.7,weight='bold')
    line([[15,y+3.2],[178,y+3.2]],color=color,lw=1.0)
    text(178,y,tag,6.5,color=MUTED,ha='right')

def module(x,y,w,h,col,title):
    rect(x,y,w,h,fc=col,radius=2)
    text(x+w/2,y+4.3,title,6.8,weight='bold',ha='center')

def chip(x,y,w,label,col,ink=INK,h=4.8):
    rect(x,y,w,h,fc=col,ec='none',radius=1)
    text(x+w/2,y+h/2,label,6.5,weight='bold',ha='center',color=ink)

def signed_strip(x,y,w,h,values):
    a=np.asarray(values,float)
    n=np.linalg.norm(a);display=a/n if n else a
    for i,v in enumerate(display):
        rect(x+i*w/len(a),y,w/len(a),h,fc=matrix_cmap(matrix_norm(v)),ec='white',lw=.15)

corpus=json.loads((ROOT/'source_data/sensor_fingerprints.json').read_text(encoding='utf-8'))
records=corpus['fingerprints']; raw=np.asarray([r['sensor_fingerprint'] for r in records])
assert raw.shape==(540,30) and np.isfinite(raw).all() and raw.max()==0
norms=np.linalg.norm(raw,axis=1)
normalized=np.divide(raw,norms[:,None],out=np.zeros_like(raw),where=norms[:,None]>0)
matrix_cmap=LinearSegmentedColormap.from_list('signed_pressure',['#243B75','#378EB3','#85C9D4','#F8FCFC'])
matrix_norm=Normalize(-1,0)

# a. A fixed physical graph: endpoint pressure level sets the edge weights.
YSHIFT=-3
panel='a'; stage('a',8.5,'Pressure → fixed zones',GREEN,'OFFLINE')
for x,w,t in [(5,42,'Pressure level'),(51,38,'Weighted links'),(93,43,'Leiden'),(140,38,'Fixed map')]:
    module(x,15,w,42,TINT[0],t)
xy=network(14.5,24,20,17,False,False,False,1.2)
for idx,col in [(2,BLUE),(9,P2),(14,TEAL)]:
    circle(float(xy[idx,0]),float(xy[idx,1]),1.2,col,EDGE,.65)
line([[11.8,30],[17.2,27.4]],color='#A8B6A2',lw=1.1)
line([[29.1,32.8],[38,37]],color='#A8B6A2',lw=1.1)
image_asset('water_tank',6.2,22.8,7.8,9.75)
image_asset('centrifugal_pump',34.4,33.2,10.5,7)
text(43.8,27,r'$p_u(t)$',6.8,color=BLUE,ha='right')

# The already-approved PCHIP curve, expanded and with unchanged controls.
tx=np.linspace(0,1,33); trace=.55+.12*np.sin(tx*2*np.pi)+.08*np.cos(tx*4*np.pi)
curve_t=np.linspace(0,1,257); curve_values=PchipInterpolator(tx,trace)(curve_t)
assert np.allclose(PchipInterpolator(tx,trace)(tx),trace)
chart_left,chart_right,chart_floor=9,42,52.6
curve_x=chart_left+(chart_right-chart_left)*curve_t
curve_y=51.5-(curve_values-.35)*22
mean_y=51.5-(float(trace.mean())-.35)*22
polygon([[chart_left,chart_floor],*np.column_stack([curve_x,curve_y]).tolist(),[chart_right,chart_floor]],fc='#D7E7F1',ec='none')
for offset,col in [(0.045,TEAL),(-0.045,P2)]:
    extra=curve_values+offset
    line(np.column_stack([curve_x,51.5-(extra-.35)*22]).tolist(),color=col,lw=.62)
line([[chart_left,chart_floor],[chart_right,chart_floor]],color='#A7BEC9',lw=.45)
line([[chart_left,mean_y],[chart_right,mean_y]],color=GREEN,lw=.7,dashed=True)
line(np.column_stack([curve_x,curve_y]).tolist(),color=BLUE,lw=1.0)
for k in [0,8,16,24,32]:
    circle(float(chart_left+(chart_right-chart_left)*tx[k]),float(51.5-(trace[k]-.35)*22),.4,'white',BLUE,.55)
text(9,54.5,r'$\bar p_u=\mathrm{mean}_t\ p_u(t)$',7.0)
text(44,51.7,'t',6.5,color=MUTED,ha='right')

# Pressure and line widths are an illustrative graph, not a rerun of Leiden.
pressure=np.array([.88,.69,trace.mean(),.75,.43,.38,.56,.61,.50,
                   trace.mean()-.045,.65,.40,.66,.53,trace.mean()+.045])
xy=NODES*np.array([29,18])+np.array([55.5,25])
ww=np.array([(pressure[u]+pressure[v])/2 for u,v in EDGES])
scaled=np.maximum((ww-ww.min())/(ww.max()-ww.min()),.001)
for (u,v),wt in zip(EDGES,scaled):
    line([xy[u].tolist(),xy[v].tolist()],color=GREEN,lw=.45+2.3*wt)
for (xx,yy),p in zip(xy,pressure):
    circle(float(xx),float(yy),1.35,matplotlib.colormaps['YlGn'](.17+p*.65),EDGE,.4)
text(70,46.5,r'$w^0_{uv}=\frac{\bar p_u+\bar p_v}{2}$',9.0,ha='center')
for i in range(24):rect(55+i*.65,53,0.65,2,fc=matplotlib.colormaps['YlGn'](.2+i*.7/23))
text(84,54.2,'≥0.001',6.5,color=MUTED,ha='right')

text(114.5,28,r'$\mathrm{Increase}\ Q_\gamma$',8,ha='center')
micro_graph(96,36,'move'); micro_graph(110,35.7,'refine'); micro_graph(124,36,'aggregate')
for xx,ss in [(100.5,'Move'),(115,'Refine'),(129.5,'Aggregate')]:
    text(xx,46.5,ss,6.5,ha='center')
arr(107.8,39,109,39,GREEN,.5);arr(121.8,39,123,39,GREEN,.5)
line([[129.5,49],[129.5,51.5],[100.5,51.5],[100.5,49]],color=GREEN,lw=.65,arrow_end=True)
text(114.5,54.5,'Resolution γ',6.5,ha='center',color=MUTED)
network(144,24,30,20.5,labels=False,sensors=True,r=1.35)
for i,(xx,lab) in enumerate([(146,'P1'),(157,'P2'),(168,'P3')]):
    chip(xx-3,47,8,lab,PART[i])
text(159,54.5,'15 zones · 30 sensors',6.7,ha='center',weight='bold')
for xx in [47.5,89.5,136.5]:arr(xx,36,xx+3,36,GREEN,1)
arr(91.5,57.6,91.5,59.1,GREEN,1.1)

# b. Paired simulations and temporal reduction before spatial encoding.
YSHIFT=-16.5
panel='b'; stage('b',75.5,'Scenario → signed fingerprint',TEAL,'SIMULATE')
for x,w,t in [(5,42,'Scenario grid'),(51,45,'Paired simulations'),(100,37,'Signed difference'),(141,37,'Sensor order')]:
    module(x,80.5,w,39,TINT[1],t)
for rr in range(5):
    for cc in range(6):pin(9.5+cc*3.4,88.9+rr*1.8,1.35)
text(43,92.5,'30 sites',6.5,ha='right')
rates=sorted(set(float(r['leak_rate_Ls']) for r in records))
assert rates==[2,5,10,20,35,50]
for i,rate in enumerate(rates):
    xx=10+i*4.9
    droplet(xx,100.9,1.45+.34*i,matplotlib.colormaps['YlOrRd'](.22+i*.09))
    text(xx,104.8,f'{rate:g}',6.5,ha='center')
text(44,100.8,'L/s',6.5,ha='right',color=MUTED)
for cx,mult,col in [(13,.7,P3),(26,1.0,PURPLE),(39,1.2,P2)]:
    t=np.linspace(0,1,33);demand=.7+.2*np.sin(2*np.pi*t)
    line([[cx-4+8*u,112-v*mult*2.5] for u,v in zip(t,demand)],color=col,lw=.9)
    line([[cx-4,112],[cx+4,112]],color='#C3D6D9',lw=.4)
    text(cx,113.7,f'×{mult:.1f}',6.5,ha='center',color=MUTED)
text(26,116.5,'30 × 6 × 3 = 540',6.8,weight='bold',ha='center')

bx,by,bw,bh=55,93.5,32,16
baseline=.78+.055*np.sin(2*np.pi*tx)+.03*np.cos(4*np.pi*tx)
leak=baseline-.35-.025*np.sin(2*np.pi*tx)
line([[bx,by-1],[bx,by+bh],[bx+bw,by+bh]],color='#AECBD0',lw=.5)
for vals,col in [(baseline,GREEN),(leak,P2)]:
    smooth=PchipInterpolator(tx,vals)(curve_t)
    yv=by+bh-smooth*bh
    polygon([[bx,by+bh],*np.column_stack([bx+curve_t*bw,yv]).tolist(),[bx+bw,by+bh]],fc=matplotlib.colors.to_rgba(col,.075),ec='none')
    line(np.column_stack([bx+curve_t*bw,yv]).tolist(),color=col,lw=1.0)
    avg=float(np.mean(vals));yy=by+bh-avg*bh
    line([[bx,yy],[91,yy]],color=col,lw=.7,dashed=True)
    circle(91,yy,1.1,col,col,.3)
    for k in [0,8,16,24,32]:circle(bx+tx[k]*bw,by+bh-vals[k]*bh,.32,'white',col,.5)
paired_means=[by+bh-float(np.mean(v))*bh for v in [baseline,leak]]
line([[91,paired_means[0]],[93,paired_means[0]],[93,paired_means[1]],[91,paired_means[1]]],color=TEAL,lw=.6)
text(55,90.5,'Baseline',6.5,color=GREEN)
text(94,90.5,'Leak',6.5,color='#B65C57',ha='right')
text(55,113,'0',6.5,color=MUTED);text(87,113,'24 h',6.5,color=MUTED,ha='right')
text(73.5,117,'Same demand state',6.5,ha='center',color=MUTED)

# Spatial bars are illustrative means; the full actual corpus is in c.
for yy,col,level,lab in [(94,P2,1,r'$\bar{\mathbf{p}}^{\,j}$'),(101,GREEN,0,r'$\bar{\mathbf{p}}^{\,0,d_j}$')]:
    for i in range(30):
        h=2.2+.9*np.sin(i*.6)**2 + (0 if level else 1.1)
        rect(105+i*.70,yy-h,.54,h,fc=col)
    text(134,yy-1.3,lab,6.6,ha='right')
text(103,98,'−',9,ha='center',color=MUTED)
line([[104,103],[132.5,103]],color='#A0C8C8',lw=.65)
signed_strip(105,105,27,2,raw[0])
text(118.5,112.2,r'$\mathbf{f}_j=\mathcal{C}(\bar{\mathbf{p}}^{\,j}-\bar{\mathbf{p}}^{\,0,d_j})$',7.2,ha='center')
text(118.5,116.5,'Threshold · round',6.5,ha='center',color=MUTED)

image_asset('pressure_sensor',146.5,89,11.18,13)
network(161,89.5,14,12,labels=False,sensors=True,r=.75)
line([[155.5,101.5],[155.5,104],[169,104]],color='#81AEBB',lw=.65)
signed_strip(145,106,29,3.8,raw[0])
text(145,113,r'$s_1$',6.5);text(174,113,r'$s_{30}$',6.5,ha='right')
text(159.5,117,'30 ordered channels',6.5,ha='center',weight='bold')
for xx in [47.5,96.5,137.5]:arr(xx,100,xx+3,100,TEAL,1)
arr(91.5,120.2,91.5,121.7,TEAL,1.1)

# c. Bind metadata and label to a numerical fingerprint; show the entire corpus.
YSHIFT=-24
panel='c';stage('c',132,'Build the fingerprint library',GOLD,'STORE')
module(5,137,42,39,TINT[2],'One record')
module(51,137,84,39,TINT[2],'540 templates × 30 sensors')
module(141,137,37,39,'#F7F6F9','Optional context')
image_asset('knowledge_layers',7.5,144.5,13.2,12)
pin(24,149,2.6);droplet(32,149,2.4,P3);wave(37,149,6,2,PURPLE)
for xx,val in [(24,records[0]['leak_node']),(32,f"{records[0]['leak_rate_Ls']:g}"),(40,f"×{records[0]['demand_multiplier']:.1f}")]:
    text(xx,154.5,val,6.5,ha='center')
signed_strip(9,158,34,4,raw[0])
text(10,168,r'$\mathbf{f}_j$',8)
line([[17,168],[22,168]],color=GOLD,lw=.7)
chip(23,165.5,8,'P0',PURPLE)
text(35,168,r'$z_j$',7.4)
text(26,173,'Fingerprint + label',6.5,ha='center',color=MUTED)
arr(47.5,156,50.5,156,GOLD,1)

# All 540 fingerprints, reflowed losslessly into 6 columns of 90 rows.
# A thin outline encloses each complete 30-channel fingerprint, not every cell.
matrix_layout=dict(blocks=6,rows_per_block=90,channels=30,block_width_mm=10.5,
                   gap_mm=1.4,height_mm=24.1,row_border_pt=.07,
                   row_border_color='#8BB1BA',order='blocks left-to-right; rows top-to-bottom')
row_h=matrix_layout['height_mm']/matrix_layout['rows_per_block']
displayed_rows=[]
for block in range(matrix_layout['blocks']):
    x=55+block*(matrix_layout['block_width_mm']+matrix_layout['gap_mm'])
    start=block*matrix_layout['rows_per_block'];stop=start+matrix_layout['rows_per_block']
    displayed_rows.extend(range(start,stop))
    text(x+5.25,144,f'{start+1}–{stop}',6.5,ha='center',color=MUTED)
    z=add(dict(type='matrix',x=x,y=145.8+YSHIFT,w=10.5,h=24.1,shape=[90,30],
               source='source_data/sensor_fingerprints.json',row_indices=[start,stop-1],
               transform='row L2 normalization; retain zero rows'))
    ax.imshow(normalized[start:stop],extent=(x,x+10.5,169.9+YSHIFT,145.8+YSHIFT),
              origin='upper',aspect='auto',cmap=matrix_cmap,norm=matrix_norm,
              interpolation='nearest',zorder=z)
    for row in range(90):
        rect(x,145.8+row*row_h,10.5,row_h,ec=matrix_layout['row_border_color'],
             lw=matrix_layout['row_border_pt'],fingerprint_row=start+row)
assert displayed_rows==list(range(540)), 'Every source fingerprint must occur exactly once'
text(90,173,'30 channels / row · 90 rows / block',6.5,ha='center',color=MUTED)
for i in range(60):rect(128,146+i*24/60,2,24/60,fc=matrix_cmap(matrix_norm(-i/59)))
text(132,146,'0',6.5,ha='left',color=MUTED)
text(131,170,'−1',6.5,ha='left',color=MUTED)

# Side branch: optional graph/text resources, not a retrieval filter.
line([[138,141],[138,173]],color='#C2B7D7',lw=.65,dashed=True)
pts=np.array([[147,147],[158,146],[168,151],[152,154],[161,158],[173,146],[146,159],[173,159]])
for u,v in [(0,1),(0,3),(1,2),(1,3),(2,4),(3,4),(1,5),(5,2),(3,6),(4,7),(2,7)]:line([pts[u].tolist(),pts[v].tolist()],color='#A49AB7',lw=.75,dashed=((u,v) in [(0,3),(2,7)]))
for i,(xx,yy) in enumerate(pts):circle(float(xx),float(yy),1.6,[P1,P2,P3,P1,PURPLE,P2,P1,P3][i],EDGE,.4)
image_asset('document',143.5,162.5,6,8)
arr(151,166.5,156,166.5,PURPLE,.7)
for j,col in enumerate([P3,PURPLE,P2,TEAL,GOLD,PURPLE,TEAL,P2]):rect(157+2.2*j,164,1.6,3+1.6*np.cos(j)**2,fc=col,radius=.4)
text(159.5,173,'Graph + text',6.5,ha='center',color=MUTED)
arr(91.5,176.7,91.5,178.2,GOLD,1.1)

# d. Amplitude fitting, residual comparison and distinct-zone ranking.
YSHIFT=-19.5
panel='d';stage('d',176.5,'Fit → match → rank zones',PURPLE,'QUERY')
rect(5,181.5,173,51.8,fc=TINT[3],radius=2)
text(91.5,186,r'$a_j=\frac{\mathbf{f}_j^{\mathsf{T}}\mathbf{y}}{\mathbf{f}_j^{\mathsf{T}}\mathbf{f}_j}$     $r_j=\|\mathbf{y}-a_j\mathbf{f}_j\|_2$',9.1,ha='center')
for xx in [41.5,130]:line([[xx,191.8],[xx,231]],color='#D7CDE7',lw=.6)

text(23,194,'30-channel query',6.7,weight='bold',ha='center')
rect(9,197,29,5.0,fc='white',ec='#ADBCD8',lw=.6,radius=.9)
signed_strip(10.5,198,21,3,raw[1]);text(35,199.5,r'$\mathbf{y}$',8,ha='center',color=BLUE)
# Actual stored vector, in fixed sensor order; not a time curve or 2-channel reduction.
profile_x=np.linspace(12,36,30);profile_zero=204.5;mm_per_m=4.0
line([[11.5,profile_zero],[36.5,profile_zero]],color='#AFBCD1',lw=.55)
for xx,value in zip(profile_x,raw[1]):
    end=profile_zero-float(value)*mm_per_m
    line([[float(xx),profile_zero],[float(xx),end]],color=BLUE,lw=.6)
    circle(float(xx),end,.27,BLUE,BLUE,.2)
text(9.8,204.5,'0',6.5,ha='right',color=MUTED)
text(9.8,212.5,'−2',6.5,ha='right',color=MUTED)
text(36.5,213.8,'m',6.5,ha='right',color=MUTED)
text(23,216,'Sensor channel',6.5,ha='center',color=MUTED)
text(23,219,'Nonzero templates',6.5,ha='center',weight='bold')
template_indices=[0,216,252,288]
for i,record_index in enumerate(template_indices):
    rr=records[record_index];zone=int(rr['leak_partition']);col=[PURPLE,P1,P2,P3][zone]
    assert zone==i and norms[record_index]>0
    yy=221.3+i*2.6
    chip(8,yy,7,f'P{zone}',col,h=2.25)
    signed_strip(16,yy,21,2.25,raw[record_index])
line([[38,199.5],[39.8,199.5],[39.8,190],[60,190],[60,186]],color=PURPLE,lw=.7)
line([[37.6,225.2],[39.8,225.2],[39.8,199.5]],color=PURPLE,lw=.7)
circle(39.8,199.5,.5,PURPLE,PURPLE,.2)
arr(60,186,68,186,PURPLE,.8)

query=np.array([-2.1,-3.9]);fa=np.array([-1.,-2.]);fb=np.array([-3.,-3.])
gain_a=float(fa@query/(fa@fa));gain_b=float(fb@query/(fb@fb))
res_a=query-gain_a*fa;res_b=query-gain_b*fb
assert np.linalg.norm(query-fa)>np.linalg.norm(query-fb)
assert np.linalg.norm(res_a)<np.linalg.norm(res_b)
assert abs(fa@res_a)<1e-12 and abs(fb@res_b)<1e-12
response_marks=[]
def response_pair(x,ybase,values,col,label,state):
    scale=2.8
    for k,(v,ref) in enumerate(zip(values,query)):
        xx=x+k*7.4;h=float(-v)*scale;hr=float(-ref)*scale
        rgb=np.array(matplotlib.colors.to_rgb(col));pale=matplotlib.colors.to_hex(rgb*.65+.35);dark=matplotlib.colors.to_hex(rgb*.87)
        rect(xx,ybase-h,4.9,h,fc=col,radius=.6)
        rect(xx+.45,ybase-h+.25,.5,max(.1,h-.5),fc=pale,radius=.18)
        rect(xx+4.1,ybase-h+.25,.35,max(.1,h-.5),fc=dark,radius=.12)
        if state=='fitted':
            z=add(dict(type='residual_gap',x=xx,y=ybase-max(h,hr)+YSHIFT,w=4.9,h=abs(h-hr),candidate=label,channel=k+1))
            ax.add_patch(Rectangle((xx,ybase-max(h,hr)+YSHIFT),4.9,abs(h-hr),fc='none',ec='#815A64',lw=0,hatch='////',zorder=z))
        rect(xx-.15,ybase-hr,5.2,hr,fc='none',ec=BLUE,lw=.8,radius=.6)
        line([[xx-.5,ybase],[xx+5.4,ybase]],color='#BBC6D8',lw=.5)
        response_marks.append(dict(candidate=label,state=state,channel=k+1,signed_value=float(v),query_signed_value=float(ref),height_mm=h,query_height_mm=hr,mm_per_m=scale))
text(86,191.5,'2-channel illustration',6.5,color=MUTED,ha='center')
text(117,191.5,'Hatch: residual',6.5,color=MUTED,ha='center')
for xx,lab in [(50,'Zone'),(67,'Stored'),(91,'Fitted'),(116,'Error (m)')]:text(xx,197,lab,6.5,weight='bold',ha='center')
for yy,lab,f,a,col,res,zlab in [(200,'A',fa,gain_a,P2,res_a,'P2'),(216,'B',fb,gain_b,P1,res_b,'P1')]:
    rect(44.5,yy,83.5,14.5,fc='white',radius=1.3)
    chip(46,yy+3.1,8,zlab,col)
    text(50,yy+11.3,lab,6.5,ha='center',color=MUTED)
    response_pair(61.5,yy+12.7,f,col,lab,'stored')
    response_pair(85.5,yy+12.7,a*f,col,lab,'fitted')
    arr(76,yy+8.5,82,yy+8.5,PURPLE,.7)
    text(79,yy+3.3,f'×{a:.2f}',6.5,color=MUTED,ha='center')
    text(109,yy+7.3,f'{np.linalg.norm(query-f):.2f}',6.5,color=MUTED,ha='center')
    arr(113.5,yy+7.3,117,yy+7.3,PURPLE,.6)
    chip(118,yy+4.8,8.6,f'{np.linalg.norm(res):.2f}','#D7EEE7' if lab=='A' else '#F0EDE3',h=5)

text(154.5,194,'Sort residual ↑',6.7,weight='bold',ha='center')
for i,k in enumerate([1,1,0,2]):
    xx=133+i*10.6
    chip(xx,200,8.2,f'P{k+1}',PART[k] if i!=1 else '#E6E2EB')
    if i==1:line([[xx+1,204.1],[xx+7.2,200.7]],color=MUTED,lw=.8)
arr(152.9,205.8,152.9,209.5,PURPLE,.8)
for i,k in enumerate([1,0,2]):chip(133+i*10.6,211,8.2,f'P{k+1}',PART[k])
arr(144.5,217.1,144.5,223,PURPLE,.7)
text(133,227,'Top-1',6.5,color=MUTED)
chip(143,224.5,8.2,'P2',P2)
network(156.5,218.5,19,13.5,labels=False,r=.9,highlight=1)

YSHIFT=0
panel='footer'
text(5,218.2,'a–b: schematics · c: simulated corpus · d: 2-channel illustration',6.5,color=MUTED)
text(178,218.2,'d: height = −Δp; blue outline = query',6.5,color=MUTED,ha='right')

# Native vector text/paths, original PNG assets and a lossless scientific matrix.
manifest=dict(canvas=dict(width_mm=W,height_mm=H),font='Arial',coordinate_origin='top-left',elements=elements,hybrid_artwork='Native vector mechanics + original Blender assets + full corpus heatmap')
(ROOT/'notes/layout_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
fig.canvas.draw(); renderer=fig.canvas.get_renderer();bounds=[]
for e,t in texts:
    b=t.get_window_extent(renderer).transformed(ax.transData.inverted())
    xx=sorted([float(b.x0),float(b.x1)]);yy=sorted([float(b.y0),float(b.y1)])
    bounds.append(dict(text=e['text'],size_pt=e['size'],bounds_mm=[xx[0],yy[0],xx[1],yy[1]],outside=xx[0]<0 or xx[1]>W or yy[0]<0 or yy[1]>H))
assert not any(b['outside'] for b in bounds),[b for b in bounds if b['outside']]
overlaps=[]
for i,b in enumerate(bounds):
    for c in bounds[i+1:]:
        a=b['bounds_mm'];d=c['bounds_mm'];w=min(a[2],d[2])-max(a[0],d[0]);h=min(a[3],d[3])-max(a[1],d[1])
        if w>.1 and h>.1:overlaps.append(dict(a=b['text'],b=c['text'],intersection_mm=[w,h]))
pdf_path=ROOT/'figures/Fig2_methodology.pdf';svg_path=ROOT/'figures/Fig2_methodology.svg'
fig.savefig(pdf_path,dpi=600);fig.savefig(svg_path,dpi=600)
svg=svg_path.read_text(encoding='utf-8').replace('image-rendering:crisp-edges;image-rendering:pixelated','image-rendering:auto')
svg_path.write_text(svg,encoding='utf-8')
reader=PdfReader(pdf_path);writer=PdfWriter();writer.clone_document_from_reader(reader)
for pg in writer.pages:
    for ref in pg['/Resources'].get('/XObject',{}).values():
        im=ref.get_object()
        if im.get('/Subtype')=='/Image':
            # Smooth Blender assets; preserve the exact 540x30 heatmap cells.
            im[NameObject('/Interpolate')]=BooleanObject(not (.40 < im.get('/Width',1)/im.get('/Height',1) < .46))
            if '/SMask' in im:im['/SMask'].get_object()[NameObject('/Interpolate')]=BooleanObject(True)
tmp=pdf_path.with_suffix('.pdf.tmp')
with tmp.open('wb') as f:writer.write(f)
tmp.replace(pdf_path)
for artist,img in images:
    artist.set_interpolation('lanczos');artist.set_resample(True);artist.set_filterrad(3)
fig.savefig(ROOT/'figures/Fig2_methodology.png',dpi=600)
fig.savefig(ROOT/'qa/Fig2_preview.png',dpi=300)
plt.close(fig)
calculation=dict(status='Two-channel teaching example, not experimental observations',y=query.tolist(),f_A=fa.tolist(),f_B=fb.tolist(),a_A=gain_a,a_B=gain_b,unfitted_distance_A=float(np.linalg.norm(query-fa)),unfitted_distance_B=float(np.linalg.norm(query-fb)),residual_A=res_a.tolist(),residual_B=res_b.tolist(),residual_norm_A=float(np.linalg.norm(res_a)),residual_norm_B=float(np.linalg.norm(res_b)),response_overlay_marks=response_marks,template_rank_zones=['P2','P2','P1','P3'],distinct_zone_rank=['P2','P1','P3'])
(ROOT/'source_data/projection_teaching_example.json').write_text(json.dumps(calculation,indent=2),encoding='utf-8')
(ROOT/'source_data/baseline_trace_display.json').write_text(json.dumps(dict(status='Schematic',control_t=tx.tolist(),control_values=trace.tolist(),mean=float(trace.mean()),display_t=curve_t.tolist(),display_values=curve_values.tolist(),interpolation='PCHIP retains all 33 controls',display_transform='y_mm = 48.5 - (value - 0.35) * 22',mean_y_mm=mean_y-3,fill='decorative, not uncertainty',additional_schematic_traces=[dict(node_index=9,offset=-.045,color=P2),dict(node_index=14,offset=.045,color=TEAL)],primary_schematic_node_index=2),indent=2),encoding='utf-8')
(ROOT/'source_data/schematic_geometry.json').write_text(json.dumps(dict(nodes=NODES.tolist(),edges=EDGES,groups=GROUP,pressure_levels=pressure.tolist(),edge_raw_weights=ww.tolist(),edge_scaled_weights=scaled.tolist(),note='15-node schematic reused across physical/weighted/fixed-map diagrams; not actual EXA7 topology or a new Leiden run.'),indent=2),encoding='utf-8')
(ROOT/'source_data/detailed_examples.json').write_text(json.dumps(dict(query_record_index=1,query_scenario_id=records[1]['scenario_id'],query_profile=raw[1].tolist(),query_profile_unit='m',profile_scale_mm_per_m=mm_per_m,template_record_indices=template_indices,template_partitions=[records[i]['leak_partition'] for i in template_indices],nominal_leak_rates_Ls=rates,demand_multipliers=[.7,1.,1.2],demand_note='Each multiplier applies to the complete 24-h baseline demand scenario.',location_glyphs=30,residual_hatch_note='Difference between fitted-template and query heights in the unchanged independent two-channel example.'),indent=2),encoding='utf-8')
qa=dict(canvas_mm=[W,H],objects=len(elements),text_objects=len(texts),word_count=sum(len(e.get('pptx_text',e['text']).split()) for e in elements if e['type']=='text'),min_font_pt=min(b['size_pt'] for b in bounds),text_bounds=bounds,text_overlap_candidates=overlaps,corpus=dict(shape=list(raw.shape),nonzero=int(np.sum(norms>0)),zero_rows=int(np.sum(norms==0)),raw_range=[float(raw.min()),float(raw.max())],display_range=[-1,0],normalization='row L2; zero remains zero',omitted_rows=0,omitted_columns=0,source_sha256=hashlib.sha256((ROOT/'source_data/sensor_fingerprints.json').read_bytes()).hexdigest()),teaching_calculation=calculation)
(ROOT/'qa/composition_audit.json').write_text(json.dumps(qa,ensure_ascii=False,indent=2),encoding='utf-8')
(ROOT/'qa/fingerprint_borders.json').write_text(json.dumps(dict(**matrix_layout,
    full_shape=list(raw.shape),rows_displayed=displayed_rows,omitted_rows=0,
    data_changed=False,display_normalization_changed=False),indent=2),encoding='utf-8')
print(json.dumps({k:qa[k] for k in ['canvas_mm','objects','text_objects','word_count','min_font_pt','text_overlap_candidates','corpus']},ensure_ascii=False,indent=2))
