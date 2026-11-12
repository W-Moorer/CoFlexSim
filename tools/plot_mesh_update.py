"""English Times New Roman figures from complete previous/updated mesh CSVs."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'gpu-contact/cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
assert 'Times New Roman' in {f.name for f in font_manager.fontManager.ttflist}
plt.rcParams.update({'font.family':'Times New Roman','font.size':11,'svg.fonttype':'none','pdf.fonttype':42,
    'axes.spines.top':False,'axes.spines.right':False,'axes.linewidth':.7,'legend.frameon':False,
    'mathtext.fontset':'stix','figure.facecolor':'white'})
OUT=ROOT/'docs/assets';OUT.mkdir(exist_ok=True)
for late in (False,True):
    fig,axes=plt.subplots(2,2,figsize=(11.8,6.8));fig.subplots_adjust(top=.87,bottom=.10,hspace=.55,wspace=.29)
    for i,case in enumerate(('helical','cam')):
        for j,version in enumerate(('previous','updated')):
            a=np.loadtxt(ROOT/f'results/mesh_update/{case}_{version}.csv',delimiter=',',skiprows=1)
            if late:a=a[a[:,0]>=a[-1,0]-(.05 if case=='helical' else .33)]
            style=dict(color=('#667682','#bd583a')[j],ls=('--','-')[j],lw=(1.7,1.1)[j],
                label=('Previous mesh','Updated mesh')[j],zorder=2+j)
            if j:style.update(marker='o',mfc='white',mew=.7,ms=3,markevery=max(1,len(a)//13))
            axes[i,0].plot(a[:,0],a[:,1],**style)
            axes[i,1].plot(a[:,0],a[:,5]*(1000 if case=='cam' else 1),**style)
        for j,ax in enumerate(axes[i]):
            ax.set_title('Helical gears' if case=='helical' else 'Cam and follower',loc='left',fontsize=12,pad=10)
            ax.text(-.13,1.08,chr(97+2*i+j),transform=ax.transAxes,fontsize=14,fontweight='bold')
            ax.set_xlabel('Time (s)');ax.grid(axis='y',color='#e5e7eb',lw=.6)
        axes[i,0].set_ylabel('Contact force (N)')
        axes[i,1].set_ylabel('Transmission error (rad)' if case=='helical' else 'Follower displacement (mm)')
    h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='upper center',ncol=2,handlelength=3,columnspacing=3)
    name='mesh_response_late' if late else 'mesh_response_full'
    fig.savefig(OUT/(name+'.png'),dpi=200,bbox_inches='tight');fig.savefig(OUT/(name+'.svg'),bbox_inches='tight');plt.close(fig)
fig,axes=plt.subplots(1,2,figsize=(11.8,3.9));fig.subplots_adjust(top=.83,bottom=.18,wspace=.3)
a=np.loadtxt(ROOT/'results/spur_native_cuda.csv',delimiter=',',skiprows=1)
for i,col in enumerate((1,5)):
    axes[i].plot(a[:,0],a[:,col],color='#bd583a',lw=1.0)
    axes[i].set_xlabel('Time (s)');axes[i].set_ylabel(('Contact force (N)','Transmission error (rad)')[i]);axes[i].grid(axis='y',color='#e5e7eb',lw=.6)
    axes[i].text(-.13,1.05,chr(97+i),transform=axes[i].transAxes,fontweight='bold',fontsize=14)
fig.suptitle('Fine-mesh spur gears | C++/CUDA',fontsize=13)
fig.savefig(OUT/'spur_response.png',dpi=200,bbox_inches='tight');fig.savefig(OUT/'spur_response.svg',bbox_inches='tight');plt.close(fig)
print('Mesh-response figures exported; timings and memory remain tables.')
