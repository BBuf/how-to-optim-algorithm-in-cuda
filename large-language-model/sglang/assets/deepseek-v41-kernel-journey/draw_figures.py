"""Original diagrams with a shared editorial theme; measured data stay unchanged."""
from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.ticker import FuncFormatter
import numpy as np

OUT=Path(__file__).resolve().parent
BG='#faf9f5'; INK='#24231f'; MUTED='#706f66'; LINE='#deddd5'
ORANGE='#bd6548'; LILAC='#8482b7'; OLIVE='#697e55'
SAND='#eeece2'; PALE_ORANGE='#f1dfd4'; PALE_LILAC='#e5e3f0'; PALE_GREEN='#e2e7da'
fm.fontManager.addfont('/System/Library/Fonts/Supplemental/Arial Unicode.ttf')
TITLE=fm.FontProperties(fname='/System/Library/Fonts/Supplemental/Songti.ttc')
plt.rcParams.update({'font.family':'Arial Unicode MS','font.size':13,'text.color':INK,
    'axes.unicode_minus':False,'figure.facecolor':BG,'axes.facecolor':BG,
    'axes.labelcolor':MUTED,'xtick.color':MUTED,'ytick.color':MUTED,'svg.fonttype':'path'})
figures=[]

def page(title,subtitle,h=650):
    fig=plt.figure(figsize=(10,h/100))
    ax=fig.add_axes([0,0,1,1]);ax.set(xlim=(0,1000),ylim=(h,0));ax.axis('off')
    txt(ax,45,35,title,24,font=TITLE)
    if subtitle:txt(ax,45,87,subtitle,12,color=MUTED)
    return fig,ax

def txt(ax,x,y,s,size=13,color=INK,ha='left',va='top',font=None):
    return ax.text(x,y,s,fontsize=size,color=color,ha=ha,va=va,fontproperties=font,
                   linespacing=1.55)

def panel(ax,x,y,w,h,fc=SAND,ec='none'):
    p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0,rounding_size=10',
                    facecolor=fc,edgecolor=ec,lw=.8)
    ax.add_patch(p);return p

def box(ax,x,y,w,h,heading,detail='',fc=SAND):
    panel(ax,x,y,w,h,fc)
    if detail:
        txt(ax,x+w/2,y+20,heading,15,ha='center')
        txt(ax,x+w/2,y+56,detail,12,ha='center',color=MUTED)
    else:txt(ax,x+w/2,y+h/2,heading,14,ha='center',va='center')

def arrow(ax,x1,y1,x2,y2,c=MUTED,rad=0):
    ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='->',lw=1.25,
        color=c,mutation_scale=15,shrinkA=4,shrinkB=4,connectionstyle=f'arc3,rad={rad}'))

def save(fig,name):
    fig.savefig(OUT/f'{name}.png',dpi=200,facecolor=BG)
    svg = OUT/f'{name}.svg'
    fig.savefig(svg,facecolor=BG)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig);figures.append(name)

# Plot the representative measurements shown in the article.
data=json.loads((OUT/'figure-data.json').read_text())
values=data['displayed_journey']
fig,base=page('从 35 到 764 tokens/s','DeepSeek-V4.1 Flash · 四卡 Blackwell · TP4 / EP4',h=1020)
for top,ht,key,color,lim,ticks,label in [
    (155,225,'bs1',ORANGE,900,[0,200,400,600,800],'BS = 1 · 输出 tokens/s'),
    (455,225,'bs64',LILAC,16000,[0,4000,8000,12000,16000],'BS = 64 · 总输出 tokens/s')]:
    txt(base,75,top-33,label,15,color=color)
    ax=fig.add_axes([.08,1-(top+ht)/1020,.86,ht/1020])
    for spine in ax.spines.values():spine.set_visible(False)
    ax.set_axisbelow(True);ax.grid(axis='y',color=LINE,lw=.65)
    ax.set_ylim(0,lim);ax.set_yticks(ticks);ax.set_xlim(.8,15.2)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x:,.0f}'))
    ax.set_xticks(range(1,16));ax.tick_params(length=0,labelsize=10.8,pad=8)
    v=values[key];xx=np.arange(1,16)
    ax.plot(xx,v,lw=2,color=color,marker='o',ms=4.3,mfc=BG,mew=1.4)
    ax.scatter([15],[v[-1]],s=36,color=color,zorder=5)
    for i,n in enumerate(v):
        text=f'{n:.1f}' if key=='bs1' else f'{n:,.0f}'
        ax.annotate(text,(i+1,n),textcoords='offset points',xytext=(0,11 if i%2==0 else -20),
                    ha='center',fontsize=10.8,color=color)
labels=['Baseline','MXFP8 GEMM','RoPE + FP4 融合','mHC 行 tile','Reduce + Sinkhorn',
        '共享 scratch','C2 池化融合','mHC 计算重叠','默认启用优化','GEMV / norm / Engram',
        '启用 DSpark','Verify mHC / WO-A','Verify kernel / mask','MoE router / 量化重叠','MoE finalize / 通信融合']
for i,label in enumerate(labels):
    col,row=divmod(i,5);x=48+col*307;y=732+row*35
    txt(base,x,y,f'{i+1:02d}',11.5,color=ORANGE)
    txt(base,x+34,y,label,11.5)
base.plot([45,950],[926,926],c=LINE,lw=.8)
txt(base,45,945,'最终 4×B300',13,color=MUTED)
txt(base,255,942,'BS=1  764 tokens/s',17,color=ORANGE)
txt(base,585,942,'BS=64  13,473 tokens/s',17,color=LILAC)
save(fig,'01-throughput-journey')

# Combine CED and KV ownership so the article needs only one architecture figure.
fig,ax=page('CED 与共享 KV cache','输入主要计算前 20 层；生成 token 仍经过完整的 40 层。',h=710)
box(ax,45,142,415,136,'Causal encoder · 20 层','处理长 prompt\n产生 decoder 使用的全局 KV',PALE_LILAC)
box(ax,540,142,415,136,'Decoder · 20 层','读取共享全局 KV\n最近 128 token 补齐局部窗口',PALE_ORANGE)
arrow(ax,464,205,536,205)
txt(ax,45,320,'全局缓存只在 4 层生成',17,font=TITLE)
# Four physical producers, instead of an unreadably dense forty-layer strip.
for x,layer,rate in [(45,'第 2 层','2 token → 1 项'),(280,'第 8 层','2 token → 1 项'),
                      (515,'第 14 层','2 token → 1 项'),(750,'第 20 层','1 token → 1 项')]:
    box(ax,x,367,205,101,layer,rate,PALE_LILAC if layer!='第 20 层' else PALE_ORANGE)
for x in [147,382,617,852]:arrow(ax,x,472,x,514)
box(ax,45,522,910,67,'其他层共享 KV；部分层重新选索引，最终读取 Top-512 全局位置',fc=SAND)
txt(ax,45,621,'FP4：主 KV 288 B + indexer K 68 B',13,color=MUTED)
txt(ax,45,655,'每个 token 的全局 KV：3514 B → 890 B，约为 V4 Flash 的 1/4',17,color=ORANGE)
save(fig,'02-architecture-cache-ownership')

fig,ax=page('KV cache 的 890 bytes/token','官方 FP4 格式下的全局 KV 逻辑大小。',h=590)
box(ax,45,145,425,150,'主 KV · 512 维','256 B 数据 + 32 B scale\n288 bytes',PALE_ORANGE)
box(ax,530,145,425,150,'Indexer K · 128 维','64 B 数据 + 4 B scale\n68 bytes',PALE_LILAC)
txt(ax,500,213,'+',22,ha='center',va='center')
txt(ax,500,338,'(288 + 68) × (3 / 2 + 1) = 890 B',24,ha='center',font=TITLE)
txt(ax,500,392,'3 份 ratio 2 缓存，加上 1 份 ratio 1 缓存',13,ha='center',color=MUTED)
for y,label,v,c in [(460,'V4 Flash',3514,'#bebcb3'),(510,'V4.1 Flash',890,ORANGE)]:
    txt(ax,45,y,label,13,va='center')
    ax.add_patch(Rectangle((215,y-13),v/3514*595,26,fc=c,ec='none'))
    txt(ax,230+v/3514*595,y,f'{v:,} B',13,va='center',color=c if v==890 else INK)
save(fig,'03-kv-bytes')

fig,ax=page('Single-Pass mHC 的计算重叠','输入混合使用前一个 sublayer 的系数，当前统计量可以并行计算。',h=510)
box(ax,45,232,150,95,'四条残差流\n当前输入',fc=SAND)
box(ax,267,149,235,96,'pre 混合','使用前一次 pre 系数',PALE_LILAC)
box(ax,570,149,217,96,'Attention / MoE',fc=PALE_LILAC)
box(ax,267,333,520,96,'统计量 + Sinkhorn','计算本次 post / comb 和下一次 pre',PALE_ORANGE)
box(ax,843,232,112,95,'post 混合',fc=SAND)
arrow(ax,195,262,267,197);arrow(ax,502,197,570,197);arrow(ax,787,197,843,258)
arrow(ax,195,296,267,381);arrow(ax,787,381,843,303)
txt(ax,520,274,'两个 stream 并行，在 post 前汇合',12.5,ha='center',color=MUTED)
txt(ax,45,464,'减少关键路径上的等待；同样适用于普通 decode、target verify 和 draft。',12.5,color=MUTED)
save(fig,'04-mhc-overlap')

fig,ax=page('DSpark：一次验证多个 token','官方 checkpoint 自带 draft 权重；本文固定 block size 5，使用真实接受结果。',h=565)
box(ax,45,145,247,142,'主模型隐藏状态','来自后面几层',SAND)
box(ax,362,145,277,142,'3 个 draft block','并行预测 5 个位置\nMarkov head 处理依赖',PALE_LILAC)
box(ax,709,145,246,142,'Target verify','批量验证\n提交可接受前缀',PALE_ORANGE)
arrow(ax,292,216,362,216);arrow(ax,639,216,709,216)
for i in range(6):
    box(ax,126+i*128,326,108,56,'锚点' if i==0 else f'draft {i}',fc=SAND if i==0 else PALE_LILAC)
txt(ax,500,403,'verify 的 token 行数 ≠ 请求 batch size',15,ha='center')
box(ax,45,453,425,65,'BS = 1 → M ≤ 6',fc=PALE_ORANGE)
box(ax,530,453,425,65,'BS = 64 → M ≤ 384',fc=PALE_ORANGE)
save(fig,'05-dspark-shapes')

fig,ax=page('MoE finalize 与通信融合','减少专家结果的中间写回和独立 kernel launch。',h=500)
txt(ax,45,144,'融合前',15,color=MUTED)
for x,w,s in [(45,205,'专家带权归约'),(295,180,'临时结果写回'),(520,185,'Shared add'),(750,205,'All-reduce')]:
    box(ax,x,185,w,84,s,fc=SAND)
for a,b in [(250,295),(475,520),(705,750)]:arrow(ax,a,227,b,227)
txt(ax,45,313,'融合后',15,color=ORANGE)
box(ax,45,354,910,91,'带权归约 → Shared add → 直接写入通信缓冲区 → 跨卡归约',fc=PALE_ORANGE)
save(fig,'06-moe-fusion')

fig,ax=page('DSpark 的 GPU 耗时','4×B300 · BS=1 · 同条件下的优化前后对比',h=460)
labels=['Target verify','Draft','完整 GPU 周期'];before=[9.049,.833,10.050];after=[6.927,.719,7.810]
for i,label in enumerate(labels):
    x=45+i*310
    panel(ax,x,145,290,242,SAND)
    txt(ax,x+145,171,label,15,ha='center')
    txt(ax,x+145,222,f'{before[i]:.3f} ms',18,ha='center',color=MUTED)
    txt(ax,x+145,270,'↓',19,ha='center',color=MUTED)
    txt(ax,x+145,318,f'{after[i]:.3f} ms',24,ha='center',color=ORANGE)
txt(ax,45,416,'Target graph 的 kernel 数：2209 → 1965',13,color=MUTED)
save(fig,'07-profile-evidence')

manifest_path=OUT/'figure-manifest.json'
manifest=json.loads(manifest_path.read_text())
manifest.update(figures=figures,generated_from='draw_figures.py',article_figures=[
    '01-throughput-journey','02-architecture-cache-ownership','04-mhc-overlap','05-dspark-shapes'],
    style={'background':BG,'ink':INK,'accents':[ORANGE,LILAC],
           'reference':'https://www.anthropic.com/engineering/multi-agent-research-system',
           'layout':'1000 px nominal width, 2x PNG; serif titles, sans-serif labels'})
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('Regenerated 7 figures; article uses 4.')
