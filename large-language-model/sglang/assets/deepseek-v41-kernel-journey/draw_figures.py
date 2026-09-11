"""Original diagrams with a shared editorial theme; measured data stay unchanged."""
from pathlib import Path
import argparse
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
parser=argparse.ArgumentParser()
parser.add_argument('--only', nargs='*', help='Regenerate only the named figures.')
args=parser.parse_args()
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
    if args.only is not None and name not in args.only:
        plt.close(fig);figures.append(name)
        return
    fig.savefig(OUT/f'{name}.png',dpi=200,facecolor=BG)
    svg = OUT/f'{name}.svg'
    fig.savefig(svg,facecolor=BG)
    svg.write_text('\n'.join(line.rstrip() for line in svg.read_text().splitlines())+'\n')
    plt.close(fig);figures.append(name)

# Ordinary decode retains its measured sequence; DSpark uses the controlled random4k/1k runs.
data=json.loads((OUT/'figure-data.json').read_text())
ordinary=data['ordinary_decode']['points']
points=data['random']['dspark_points']
height=1080
fig,base=page(f"从 35 到 {points[-1]['output_tps_median']:.0f} tokens/s：kernel 优化历程",
              'SGLang · DeepSeek-V4.1 Flash · 4×GB300 · TP4 / EP4 · BS=1',h=height)
top,ht=155,250
txt(base,80,125,'输出 tokens/s',12,color=MUTED)
ax=fig.add_axes([.08,1-(top+ht)/height,.86,ht/height])
for spine in ax.spines.values():spine.set_visible(False)
ax.set_axisbelow(True);ax.grid(axis='y',color=LINE,lw=.65)
v=[p['output_tps'] for p in ordinary]+[p['output_tps_median'] for p in points]
ax.set_ylim(0,880);ax.set_yticks([0,200,400,600,800]);ax.set_xlim(.6,13.4)
ax.set_xticks(range(1,14),[f'{i:02d}' for i in range(1,14)])
ax.tick_params(length=0,labelsize=11,pad=9)
ax.plot(range(1,11),v[:10],lw=2.2,color=LILAC,marker='o',ms=5,mfc=BG,mew=1.6)
ax.plot([10,11],v[9:11],lw=1.5,color=MUTED,ls=(0,(3,3)))
ax.plot(range(11,14),v[10:],lw=2.4,color=ORANGE,marker='o',ms=6,mfc=BG,mew=1.8)
for i,n in enumerate(v):
    offset=12 if i>=10 or i%2==0 else -20
    ax.annotate(f'{n:.1f}',(i+1,n),textcoords='offset points',xytext=(0,offset),
                ha='center',fontsize=12 if i>=10 else 11.2,color=ORANGE if i>=10 else LILAC)
ax.text(1.1,720,'普通 decode → DSpark → Verify / MoE → 小 batch 融合',fontsize=12.5,color=INK)
ax.text(1.1,605,'DSpark：随机 4k/1k，模拟 accept length = 5.5',fontsize=12,color=MUTED)

txt(base,45,463,'01—10  普通 decode',17,color=LILAC,font=TITLE)
txt(base,955,469,'累计输出速度 · tokens/s',11.5,color=MUTED,ha='right')
for i,p in enumerate(ordinary):
    col,row=divmod(i,5);x=45+col*465;y=511+row*36
    txt(base,x,y,f'{i+1:02d}',12,color=LILAC)
    txt(base,x+36,y,p['label'],12.5)
    txt(base,x+425,y,f"{p['output_tps']:.1f}",12.5,ha='right',color=LILAC)

txt(base,45,716,'11—13  DSpark',17,color=ORANGE,font=TITLE)
txt(base,955,722,'相同随机输入 · 模拟 accept length 目标 5.5',11.5,color=MUTED,ha='right')
for i,p in enumerate(points):
    x=45+i*310
    panel(base,x,763,290,244,PALE_ORANGE if i!=1 else SAND)
    txt(base,x+17,781,f"{i+11:02d}  {p['label']}",14)
    txt(base,x+17,818,f"{p['output_tps_median']:.1f}",25,color=ORANGE)
    txt(base,x+128,835,'tokens/s',11,color=MUTED)
    txt(base,x+17,875,'\n'.join(p['methods']),11.5,color=MUTED)

txt(base,45,1030,'01—10 保留普通 decode 测量；11—13 为随机 4096/1024、固定模拟接受长度的测量。',11,color=MUTED)
txt(base,45,1056,'DSpark 三组实测 accept length 中位数均为 5.505。',11,color=MUTED)
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

fig,ax=page('Single-Pass mHC 的计算重叠','输入混合使用前一个 sublayer 的系数，统计量与 Attention / MoE 重叠。',h=510)
box(ax,45,232,150,95,'四条残差流\n当前输入',fc=SAND)
box(ax,267,149,235,96,'pre 混合 + RMSNorm','使用前一次 pre 系数',PALE_LILAC)
box(ax,570,149,217,96,'Attention / MoE',fc=PALE_LILAC)
box(ax,267,333,520,96,'统计量 + Sinkhorn','计算本次 post / comb 和下一次 pre',PALE_ORANGE)
box(ax,843,232,112,95,'post 混合',fc=SAND)
arrow(ax,195,262,267,197);arrow(ax,502,197,570,197);arrow(ax,787,197,843,258)
arrow(ax,195,296,267,381);arrow(ax,787,381,843,303)
txt(ax,520,274,'两个 stream 并行，在 post 前汇合',12.5,ha='center',color=MUTED)
txt(ax,45,464,'小 batch 先完成融合的 pre 混合 / RMSNorm，再启动统计量，减少短算子之间的争用。',12.5,color=MUTED)
save(fig,'04-mhc-overlap')

fig,ax=page('DSpark：一次验证多个 token','官方 checkpoint 自带 draft 权重；测量固定 block size 5、模拟 accept length 5.5。',h=565)
box(ax,45,145,247,142,'主模型隐藏状态','来自后面几层',SAND)
box(ax,362,145,277,142,'3 个 draft block','并行预测 5 个位置\nMarkov head 处理依赖',PALE_LILAC)
box(ax,709,145,246,142,'Target verify','批量验证\n提交可接受前缀',PALE_ORANGE)
arrow(ax,292,216,362,216);arrow(ax,639,216,709,216)
for i in range(6):
    box(ax,126+i*128,326,108,56,'锚点' if i==0 else f'draft {i}',fc=SAND if i==0 else PALE_LILAC)
txt(ax,500,403,'verify 的 token 行数 ≠ 请求 batch size',15,ha='center')
box(ax,245,453,510,65,'1 个请求（BS = 1）→ verify 最多处理 6 行',fc=PALE_ORANGE)
save(fig,'05-dspark-shapes')

fig,ax=page('MoE finalize 与通信融合','减少专家结果的中间写回和独立 kernel launch。',h=500)
txt(ax,45,144,'融合前',15,color=MUTED)
for x,w,s in [(45,205,'专家带权归约'),(295,180,'临时结果写回'),(520,185,'Shared add'),(750,205,'All-reduce')]:
    box(ax,x,185,w,84,s,fc=SAND)
for a,b in [(250,295),(475,520),(705,750)]:arrow(ax,a,227,b,227)
txt(ax,45,313,'融合后',15,color=ORANGE)
box(ax,45,354,910,91,'带权归约 → Shared add → 直接写入通信缓冲区 → 跨卡归约',fc=PALE_ORANGE)
save(fig,'06-moe-fusion')

manifest_path=OUT/'figure-manifest.json'
manifest=json.loads(manifest_path.read_text())
manifest.update(figures=['00-cover',*figures],generated_from='draw_figures.py',static_figures=['00-cover'],article_figures=[
    '00-cover','01-throughput-journey','02-architecture-cache-ownership','04-mhc-overlap','05-dspark-shapes'],
    style={'background':BG,'ink':INK,'accents':[ORANGE,LILAC],
           'reference':'https://www.anthropic.com/engineering/multi-agent-research-system',
           'layout':'1000 px nominal width, 2x PNG; serif titles, sans-serif labels'})
manifest_path.write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
print('Regenerated figures:', ', '.join(args.only or figures))
