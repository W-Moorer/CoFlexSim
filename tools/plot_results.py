"""Regenerate the README figures from the committed measured records."""
import sys,os,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['MPLCONFIGDIR']=str(ROOT/'gpu-contact/cache/matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
available={f.name for f in font_manager.fontManager.ttflist}
fonts=[f for f in ('Times New Roman','Microsoft YaHei','Noto Sans CJK SC','DejaVu Sans') if f in available]
if 'Times New Roman' not in fonts:raise RuntimeError('Install Times New Roman before rendering the research figures')
if not any(f in fonts for f in ('Microsoft YaHei','Noto Sans CJK SC')):raise RuntimeError('Install Microsoft YaHei or Noto Sans CJK SC')
plt.rcParams.update({'font.family':fonts,
    'font.size':11,'axes.unicode_minus':False,'svg.fonttype':'path','figure.facecolor':'white',
    'axes.spines.top':False,'axes.spines.right':False,'savefig.bbox':'tight'})
OUT=ROOT/'docs/assets';OUT.mkdir(parents=True,exist_ok=True)
QA=ROOT/'outputs/figure_qa';QA.mkdir(parents=True,exist_ok=True)
def save(fig,name):
    fig.savefig(OUT/(name+'.svg'));fig.savefig(QA/(name+'.png'),dpi=130);plt.close(fig)
labels={'spur':'细网格直齿轮','helical':'斜齿轮','cam':'凸轮–从动件'}
palette=['#315e8d','#d26336','#159181','#8662a5']
for late in (False,True):
    fig,axes=plt.subplots(3,2,figsize=(13,10),layout='constrained')
    for i,case in enumerate(labels):
        for j,backend in enumerate(('host_cuda','native_cuda')):
            a=np.loadtxt(ROOT/f'results/{case}_{backend}.csv',delimiter=',',skiprows=1)
            if late:a=a[a[:,0]>=a[-1,0]-(.33 if case=='cam' else .05)]
            title='原 GPU 流程' if case=='spur' else '优化主机调度 GPU'
            label=title if j==0 else '统一 C++/CUDA 后端'
            style=dict(color=palette[j],ls='--' if j==0 else '-',lw=1.6 if j==0 else 1.0,label=label)
            if j:style.update(marker='o',mfc='white',ms=3,markevery=max(1,len(a)//24))
            axes[i,0].plot(a[:,0],a[:,1],**style)
            axes[i,1].plot(a[:,0],a[:,5]*(1000 if case=='cam' else 1),**style)
        for ax in axes[i]:ax.grid(alpha=.18);ax.set_xlabel('物理时间 / s');ax.legend(fontsize=9,loc='upper right')
        axes[i,0].set(title=labels[case]+' · 接触合力',ylabel='合力 / N')
        axes[i,1].set(title=labels[case]+(' · 从动件坐标' if case=='cam' else ' · 传动误差'),ylabel='位移 / mm' if case=='cam' else '传动误差 / rad')
    fig.suptitle(('末段响应对比' if late else '完整长时程响应对比')+' · 同模型、同步长、无相位平移',fontsize=17)
    save(fig,'response_late' if late else 'response_full')
m=json.loads((ROOT/'results/measurements.json').read_text());f=m['fine_spur'];o=m['other_cases']['cases']
fig,axes=plt.subplots(1,3,figsize=(13,4.3),layout='constrained')
for ax,case in zip(axes,labels):
    if case=='spur':
        sets=[v['online_s'] for v in f['window']['variants'].values()];names=['原 GPU','C++/CUDA'];subtitle='200–212 ms · 每组 3 次'
    else:
        c=o['guided_cam' if case=='cam' else case];sets=[c['short'][b]['online_s'] for b in ('cpu_columns','window_gpu','host_cuda','native_cuda')]
        names=['CPU','原 GPU','优化 GPU','C++/CUDA'];subtitle=('0–330 ms' if case=='cam' else '0–12 ms')+' · 每组 3 次'
    for j,y in enumerate(sets):
        med=np.median(y);ax.bar(j,med,color=palette[j%4],width=.6,zorder=2)
        ax.scatter(np.full(len(y),j)+np.linspace(-.12,.12,len(y)),y,color='#25313e',s=15,zorder=3)
        ax.text(j,med*1.12,f'{med:.3f}',ha='center',fontsize=10)
    ax.set_xticks(range(len(names)),names,rotation=16);ax.set_yscale('log');ax.set_ylim(.5,max(max(x) for x in sets)*1.8)
    ax.set_title(labels[case]+'\n'+subtitle);ax.set_ylabel('在线耗时 / s（对数刻度）');ax.grid(axis='y',alpha=.18)
fig.suptitle('RTX 3070 · 在线求解实测（不含预计算与初始化）',fontsize=16);save(fig,'timing_short')
fig,axes=plt.subplots(1,2,figsize=(12,4.6),layout='constrained')
gpu=[];host=[]
for old,new in [('fastcapture','native_strict')]:
    gpu.append([f['memory_diagnostics'][k]['owned_device_peak_upper_bound_bytes']/2**20 for k in (old,new)])
    host.append([f['memory_diagnostics'][k]['sampled_host_rss_bytes']/2**20 for k in (old,new)])
for c in (o['helical'],o['guided_cam']):
    gpu.append([c['memory_diagnostics'][k]['owned_live_upper_bound_MiB'] for k in ('host_cuda','native_cuda')])
    host.append([c['memory_diagnostics'][k]['host_peak_RSS_MiB'] for k in ('host_cuda','native_cuda')])
for ax,data,title in zip(axes,[gpu,host],['GPU 活跃分配峰值上界','CPU 进程 RSS']):
    a=np.array(data)
    for j,label in enumerate(('参考 GPU 流程','统一 C++/CUDA')):
        bars=ax.bar(np.arange(3)+(j-.5)*.32,a[:,j],width=.3,color=palette[j],label=label)
        ax.bar_label(bars,fmt='%.1f',fontsize=9,padding=3)
    ax.set_xticks(np.arange(3),list(labels.values()));ax.set_ylabel('MiB');ax.set_title(title);ax.set_ylim(0,a.max()*1.2);ax.grid(axis='y',alpha=.18);ax.legend(fontsize=9)
fig.suptitle('独立内存诊断 · 非计时运行；直齿轮 RSS 为采样值，其余为进程峰值',fontsize=14)
save(fig,'memory')
print('Four SVG figures and Python-rendered QA previews written.')
