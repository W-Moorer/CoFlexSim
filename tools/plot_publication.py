"""Publication-style English figures from the committed response CSVs."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'gpu-contact/cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
assert 'Times New Roman' in {f.name for f in font_manager.fontManager.ttflist}, 'Install Times New Roman'
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'svg.fonttype':'none',
    'pdf.fonttype':42,'figure.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.7,'axes.labelsize':11,'axes.titlesize':12,'legend.frameon':False,
    'xtick.direction':'out','ytick.direction':'out','mathtext.fontset':'stix'})
OUT=ROOT/'docs/assets';OUT.mkdir(parents=True,exist_ok=True)
QA=ROOT/'outputs/figure_qa';QA.mkdir(parents=True,exist_ok=True)
def save(fig,name):
    fig.savefig(OUT/(name+'.svg'),bbox_inches='tight')
    fig.savefig(OUT/(name+'.png'),dpi=200,bbox_inches='tight')
    fig.savefig(QA/(name+'.png'),dpi=120,bbox_inches='tight')
    plt.close(fig)
names={'spur':'Fine-mesh spur gears','helical':'Helical gears','cam':'Cam and follower'}
for late in (False,True):
    fig,axes=plt.subplots(3,2,figsize=(11.8,8.6))
    fig.subplots_adjust(top=.89,bottom=.08,hspace=.62,wspace=.30)
    for i,(case,title) in enumerate(names.items()):
        for j,backend in enumerate(('host_cuda','native_cuda')):
            a=np.loadtxt(ROOT/f'results/{case}_{backend}.csv',delimiter=',',skiprows=1)
            if late:a=a[a[:,0]>=a[-1,0]-(.33 if case=='cam' else .05)]
            style=dict(color=('#64748b','#b35338')[j],ls=('--','-')[j],lw=(1.6,1.05)[j],
                label=('Reference GPU','C++/CUDA')[j],zorder=2+j)
            if j:style.update(marker='o',mfc='white',mew=.7,ms=3,markevery=max(1,len(a)//17))
            axes[i,0].plot(a[:,0],a[:,1],**style)
            axes[i,1].plot(a[:,0],a[:,5]*(1000 if case=='cam' else 1),**style)
        for j,ax in enumerate(axes[i]):
            ax.set_xlabel('Time (s)');ax.grid(axis='y',color='#e5e7eb',lw=.6)
            ax.set_title(title,loc='left',pad=10)
            ax.text(-.14,1.09,chr(97+i*2+j),transform=ax.transAxes,fontweight='bold',fontsize=14)
            ax.tick_params(labelsize=10)
        axes[i,0].set_ylabel('Contact force (N)')
        axes[i,1].set_ylabel('Follower displacement (mm)' if case=='cam' else 'Transmission error (rad)')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.0),ncol=2,handlelength=3,columnspacing=3)
    save(fig,'response_late' if late else 'response_full')
fig,ax=plt.subplots(figsize=(12,5.5));ax.set(xlim=(0,12),ylim=(0,5.5));ax.axis('off')
def box(x,y,w,h,text,color='#edf3f5',edge='#6b8894'):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.04,rounding_size=.07',fc=color,ec=edge,lw=.9))
    ax.text(x+w/2,y+h/2,text,ha='center',va='center',fontsize=11,linespacing=1.4)
def arrow(start,end,label=None,connection='arc3,rad=0'):
    ax.add_patch(FancyArrowPatch(start,end,arrowstyle='-|>',mutation_scale=10,lw=.9,color='#4b5563',connectionstyle=connection))
    if label:ax.text((start[0]+end[0])/2,(start[1]+end[1])/2+.11,label,ha='center',fontsize=10)
ax.text(.15,5.12,'a  Offline preparation',fontsize=13,fontweight='bold')
box(.2,3.7,2.7,1,'FE mesh, materials\nand constraints')
box(3.6,3.7,3.5,1,'Dynamic modes +\nsurface residual compliance')
box(8.0,3.7,3.6,1,'Reusable operator library\nC++ regional cache',color='#faf0e7',edge='#b88a62')
arrow((2.95,4.2),(3.55,4.2));arrow((7.15,4.2),(7.95,4.2))
ax.text(.15,2.9,'b  Online time stepping',fontsize=13,fontweight='bold')
box(.2,1.4,2.1,1,'Motion prediction\nGPU')
box(2.8,1.4,2.7,1,'Surface update\nBVH / contact geometry')
box(6.05,1.4,2.75,1,'Pressure integration\nCoupled GPU iterations')
box(9.3,1.4,2.3,1,'Accept state\nUpdate HHT history')
arrow((2.35,1.9),(2.75,1.9));arrow((5.55,1.9),(6.,1.9));arrow((8.85,1.9),(9.25,1.9))
arrow((8.2,3.65),(7.7,2.45));ax.text(7.35,3.15,'Region data',fontsize=10)
arrow((8.5,2.45),(9.,3.65));ax.text(9.15,3.48,'Refill on miss',fontsize=10)
ax.plot([10.45,10.45,1.25],[1.35,.4,.4],color='#4b5563',lw=.9)
arrow((1.25,.4),(1.25,1.35));ax.text(5.6,.08,'Next time step',ha='center',fontsize=11)
ax.text(7.4,1.02,'Geometry and equilibrium checked within iterations',ha='center',fontsize=10)
box(9.3,2.75,2.3,.55,'Field postprocessing',color='#f3f4f6',edge='#8b9198')
arrow((10.45,2.45),(10.45,2.7))
save(fig,'workflow')
print('English Times New Roman response figures and workflow exported. Performance stays in tables.')
