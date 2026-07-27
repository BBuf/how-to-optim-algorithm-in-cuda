"""渲染 SGLang custom allreduce v1/v2 博客的配图（6 张）。

用法：python3 draw_custom_allreduce.py
输出：custom_ar_fig{1..6}_*.png（240 dpi）到本目录。
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

plt.rcParams["font.family"] = ["Hiragino Sans GB", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

INK = "#1C222B"
MUTED = "#5E6875"
HAIR = "#D8D9D4"
PULL, PULL_S = "#2563A8", "#E3EDF7"
PUSH, PUSH_S = "#B4562A", "#F6E8DF"
OP, OP_S = "#3D4654", "#EEF0EC"
MC, MC_S = "#2E7D5B", "#E2F0E9"
V1, V1_S = "#6B5B9A", "#ECE7F4"

OUT = pathlib.Path(__file__).resolve().parent


def canvas(w, h, title):
    fig, ax = plt.subplots(figsize=(w, h), dpi=240)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    if title:
        ax.text(0, 99, title, fontsize=11.5, fontweight="bold", color=INK, va="top")
    return fig, ax


def box(ax, x, y, w, h, text, fc=OP_S, ec=OP, fs=9.2, color=None, sub=None,
        lw=1.1, dashed=False, bold=False, align="center"):
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
            facecolor=fc, edgecolor=ec, linewidth=lw,
            linestyle=(0, (4, 2)) if dashed else "solid", zorder=2,
        )
    )
    tx = x + w / 2 if align == "center" else x + 1.4
    ha = "center" if align == "center" else "left"
    if sub:
        ax.text(tx, y + h * 0.63, text, ha=ha, va="center", fontsize=fs,
                color=color or INK, zorder=3,
                fontweight="bold" if bold else "normal")
        ax.text(tx, y + h * 0.28, sub, ha=ha, va="center", fontsize=fs - 1.4,
                color=MUTED, zorder=3)
    else:
        ax.text(tx, y + h / 2, text, ha=ha, va="center", fontsize=fs,
                color=color or INK, zorder=3,
                fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2, color=OP, lw=1.2, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=9,
            color=color, linewidth=lw, zorder=4,
            linestyle=(0, (3, 2)) if dashed else "solid",
            shrinkA=0, shrinkB=0,
        )
    )


def note(ax, x, y, text, fs=8.2, color=MUTED, ha="left"):
    ax.text(x, y, text, fontsize=fs, color=color, va="center", ha=ha, zorder=3)


def panel(ax, x, y, w, h, title, items, fc=OP_S, ec=OP, tc=None, fs=9.4,
          item_fs=8.2, lw=1.1, dashed=False):
    """带顶部标题 + 若干条目的面板（标题不居中，避免与条目重叠）。"""
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
            facecolor=fc, edgecolor=ec, linewidth=lw,
            linestyle=(0, (4, 2)) if dashed else "solid", zorder=2,
        )
    )
    ax.text(x + 1.6, y + h - 1.9, title, fontsize=fs, fontweight="bold",
            color=tc or INK, va="center", zorder=3)
    step = (h - 4.6) / max(len(items), 1)
    for i, t in enumerate(items):
        ax.text(x + 1.6, y + h - 4.6 - i * step, t, fontsize=item_fs,
                color=MUTED, va="center", zorder=3)


def save(fig, name):
    fig.savefig(OUT / name, dpi=240, facecolor="white", bbox_inches="tight",
                pad_inches=0.16)
    plt.close(fig)
    print("rendered", name)


# --------------------------------------------------------------------------
# 图 1：all_reduce 分发决策链
# --------------------------------------------------------------------------
fig, ax = canvas(11.5, 7.4, "GroupCoordinator.all_reduce 的分发决策链（parallel_state.py）")

box(ax, 2, 86, 40, 6, "world_size == 1 → 原样返回")
box(ax, 2, 78, 40, 6, "CPU 张量 → shm_allreduce")
box(ax, 2, 70, 40, 6, "hpu / xpu / npu 平台通道")
arrow(ax, 22, 70, 22, 65)
box(ax, 2, 55, 46, 9.5, "torch.compile 追踪中", fc=OP_S, ec=OP, dashed=True,
    sub="字节阈值会 guard 符号形状 → 统一发 outplace 自定义算子，选择推迟到运行时")
arrow(ax, 22, 55, 22, 50)
box(ax, 2, 40, 46, 9.5, "--enable-symm-mem 且 pynccl 可用", fc=MC_S, ec=MC,
    sub="是（且 pymscclpp 不接）→ pynccl symm-mem 快路径（原地）")
arrow(ax, 22, 40, 22, 35)

ax.add_patch(FancyBboxPatch((2, 12), 46, 22.5,
             boxstyle="round,pad=0,rounding_size=0.8",
             facecolor=PULL_S, edgecolor=PULL, linewidth=1.5, zorder=2))
ax.text(25, 31.5, "_resolve_outplace_all_reduce_method 依序尝试", ha="center",
        fontsize=9.2, color=PULL, fontweight="bold", zorder=3)
for i, t in enumerate([
    "① ca — custom AR（should_custom_ar 为真）",
    "② qr — quick_all_reduce（仅 ROCm）",
    "③ pymscclpp　　④ torch_symm_mem",
    "⑤ piecewise-graph 下的 pynccl（outplace）",
    "命中即走出参通道（返回新张量）",
]):
    ax.text(25, 27.5 - i * 3.1, t, ha="center", fontsize=8.4,
            color=MUTED if i == 4 else INK, zorder=3)
arrow(ax, 22, 12, 22, 8)
box(ax, 2, 2, 46, 6, "兜底：inplace_all_reduce（pynccl / NCCL）")

panel(ax, 56, 52, 42, 27, "CustomAllReduceV2（默认）", [
    "torch symmetric memory 存储平面",
    "三算法：1shot_push / 1shot_pull / 2shot_pull",
    "NVLS multicast、PDL、graph 指针表",
    "JIT，per (dtype, world_size, PDL) 实例化",
    "world_size 2..8（含奇数）",
    "调优表 per (arch, ws) × (graph, eager)",
], fc=PULL_S, ec=PULL, tc=PULL)

panel(ax, 56, 25, 42, 20, "CustomAllreduce（v1，legacy）", [
    "cudaMalloc + cudaIpc 手工交换句柄",
    "两算法：1stage / 2stage（与 vLLM 同源）",
    "C++ 类持有全部状态，预编译进 sgl-kernel",
    "world_size 2/4/6/8，尺寸上限 8 MB",
], fc=V1_S, ec=V1, tc=V1)

arrow(ax, 48, 22, 56, 33, color=V1, dashed=True)
arrow(ax, 48, 28, 56, 62, color=PULL)
save(fig, "custom_ar_fig1_dispatch.png")


# --------------------------------------------------------------------------
# 图 2：v1 的 1stage 与 2stage
# --------------------------------------------------------------------------
fig, ax = canvas(11.5, 7.0, "v1 的两个算法（world_size=4，站在 rank0 视角）")

ax.text(0, 93, "cross_device_reduce_1stage", fontsize=9.8, color=V1,
        fontweight="bold", va="top")
for i in range(4):
    box(ax, 2 + i * 17, 80, 14, 6, f"rank{i} 数据", fc=V1_S, ec=V1, fs=8.6)
for i in range(4):
    arrow(ax, 9 + i * 17, 80, 34 + i * 0.6, 73, color=V1, lw=1.0)
box(ax, 22, 66, 30, 6.5, "rank0 读全部 4 份 → 累加 → 写本地输出", fs=8.8)
note(ax, 56, 71, "每个 rank 都执行同样的动作")
note(ax, 56, 68, "NVLink 读流量 = ws × N；两轮 barrier 夹一段归约")

ax.text(0, 58, "cross_device_reduce_2stage（reduce-scatter + all-gather）",
        fontsize=9.8, color=PULL, fontweight="bold", va="top")
for i, (c, cs) in enumerate([(PULL, PULL_S), (PUSH, PUSH_S), (MC, MC_S), (V1, V1_S)]):
    box(ax, 2 + i * 17, 46, 17, 6, f"shard{i} ← rank{i} 归约", fc=cs, ec=c, fs=8.0)
note(ax, 72, 49, "stage1：每 rank 读 ws 份")
note(ax, 72, 46, "自己的分片，写进自己 tmp buf")
arrow(ax, 35, 46, 35, 41)
box(ax, 16, 34, 38, 6.5, "multi_gpu_barrier（need_fence：release / acquire）",
    lw=1.5, fs=8.8)
arrow(ax, 35, 34, 35, 29)
box(ax, 8, 22, 54, 6.5, "stage2：每 rank 把 4 个分片取回，拼出完整结果", fs=8.8)
note(ax, 72, 25, "总流量 ≈ 2 × N")

box(ax, 2, 2, 96, 15, "", fc="white", ec=HAIR)
ax.text(3.5, 14, "切换规则（写死在 kernel 侧）", fontsize=9.2, fontweight="bold",
        color=INK, va="center", zorder=3)
note(ax, 3.5, 10.2,
     "ws=2 恒 1stage；full NVLink 时 ws≤4 且 <512 KB、或 ws≤8 且 <256 KB → 1stage，否则 2stage",
     fs=8.6, color=INK)
note(ax, 3.5, 6.4,
     "SGLANG_CUSTOM_ALLREDUCE_ALGO=1stage|2stage 可强制；v1 入场上限 8 MB，超过回落 NCCL",
     fs=8.2)
save(fig, "custom_ar_fig2_v1_algos.png")


# --------------------------------------------------------------------------
# 图 3：v2 的存储平面
# --------------------------------------------------------------------------
fig, ax = canvas(11.5, 6.2, "v2 的存储平面：一次 symmetric memory 分配切三段（_init_workspace）")

ax.text(0, 92, "每个 rank 一块 symmetric memory，布局相同", fontsize=9.2,
        color=INK, va="top")
box(ax, 2, 78, 52, 9, "push workspace：2 × ws 个 buffer", fc=PUSH_S, ec=PUSH,
    sub="2 相位 × 每 peer 一格，各 max_push_size", fs=8.8)
box(ax, 54, 78, 30, 9, "pull workspace", fc=PULL_S, ec=PULL,
    sub="max_pull_size，单块", fs=8.8)
box(ax, 84, 78, 14, 9, "semaphores", sub="128 B × 块数", fs=8.6)

for j, alpha in enumerate([0.55, 0.3]):
    y = 72 - j * 4.5
    note(ax, 2, y + 1.4, f"rank{j+1} 同布局", fs=7.8)
    ax.add_patch(Rectangle((16, y), 40, 2.6, facecolor=PUSH_S, edgecolor=PUSH,
                           alpha=alpha, zorder=2))
    ax.add_patch(Rectangle((56, y), 23, 2.6, facecolor=PULL_S, edgecolor=PULL,
                           alpha=alpha, zorder=2))
    ax.add_patch(Rectangle((79, y), 11, 2.6, facecolor=OP_S, edgecolor=OP,
                           alpha=alpha, zorder=2))

box(ax, 2, 53, 96, 8.5, "multicast 平面（symm_mem.multicast_ptr + pull 偏移）",
    fc=MC_S, ec=MC, dashed=True, color=MC, fs=9.0,
    sub="同一个地址：写 = 广播到所有 rank，读 = 交换机归约后返回；不可用时 num_mc_blocks 置 None")

panel(ax, 2, 32, 46, 16, "本地（非对称）张量", [
    "push_counter：[num_push_blocks] u32，push 相位计数",
    "graph_params：[131072, ws] u64 指针表（ws=8 时 8 MB）",
])
panel(ax, 52, 32, 46, 16, "C++ Communicator = 纯指针视图", [
    "TensorMatcher 校验各 rank 形状一致、uint8、连续、CUDA",
    "不做任何分配 / IPC；生命周期全部由 Python 侧持有",
])

box(ax, 2, 4, 96, 23, "", fc="white", ec=HAIR)
ax.text(3.5, 23.5, "尺寸来源", fontsize=9.2, fontweight="bold", color=INK,
        va="center", zorder=3)
for i, t in enumerate([
    "调优表给出 (arch, ws) 的 max_push / max_pull 期望值 → min(表值, 16 MB cap) → 1 KB 对齐",
    "构造参数 max_pull_size / max_push_size 可显式覆盖（push-only 的融合算子实例传极小 pull）",
    "bench / 测试可用 uncap_pull_thresholds() 把 2shot 阈值抬到 workspace 容量",
]):
    note(ax, 3.5, 19 - i * 4.2, t, fs=8.4, color=INK if i == 0 else MUTED)
save(fig, "custom_ar_fig3_v2_workspace.png")


# --------------------------------------------------------------------------
# 图 4：v2 三算法 + multicast 档
# --------------------------------------------------------------------------
fig, ax = canvas(11.5, 8.6, "v2 的三个算法与 multicast 变体档的数据流")

ax.text(0, 96, "1shot_push（超小消息，零跨 rank barrier）", fontsize=9.8,
        color=PUSH, fontweight="bold", va="top")
box(ax, 2, 85, 16, 7, "本地 input", fc=PUSH_S, ec=PUSH, fs=8.6)
arrow(ax, 18, 88.5, 24, 88.5, color=PUSH)
box(ax, 24, 84, 36, 9, "st.relaxed.sys 写入全部 ws 个 rank 的", fc=PUSH_S, ec=PUSH,
    sub="push workspace [属于我的那一格]", fs=8.8)
arrow(ax, 60, 88.5, 66, 88.5, color=PUSH)
box(ax, 66, 84, 32, 9, "自旋轮询本地 ws 个格子", sub="全部到齐 → 归约 → 写 output", fs=8.8)
note(ax, 2, 81, "唯一同步：本地 push_counter 相位计数（决定用双 buffer 的哪一半）；数据到达本身即信号")
note(ax, 2, 78, "pos_zero 哨兵：push 前把载荷中真实的 +0.0 改写为 −0.0（数值等价），poll 端见 +0.0 即继续等，消费完回填 +0.0")

ax.text(0, 73, "1shot_pull（小中消息，两轮信号量夹一段全量读归约）", fontsize=9.8,
        color=PULL, fontweight="bold", va="top")
box(ax, 2, 62, 22, 8, "eager：input 先 memcpy", fc=PULL_S, ec=PULL,
    sub="进本地 pull workspace", fs=8.4)
arrow(ax, 24, 66, 30, 66, color=PULL)
box(ax, 30, 62, 20, 8, "sync_enter", sub="打旗 + 等 ws 个旗", lw=1.5, fs=8.6)
arrow(ax, 50, 66, 56, 66, color=PULL)
box(ax, 56, 62, 24, 8, "读全部 ws 份 → 归约", fc=PULL_S, ec=PULL,
    sub="→ 写 output", fs=8.6)
arrow(ax, 80, 66, 86, 66, color=PULL)
note(ax, 86.5, 66, "sync_exit", fs=8.4)
note(ax, 2, 59, "数据源三选一（pull_arg）：eager = 本地 workspace（付进出拷贝）；graph = 指针表一行（零拷贝）；multicast 见下")

ax.text(0, 54, "2shot_pull（中大消息，reduce-scatter 与 all-gather 融合，原地写回）",
        fontsize=9.8, color=PULL, fontweight="bold", va="top")
for i, (c, cs) in enumerate([(PULL, PULL_S), (PUSH, PUSH_S), (MC, MC_S), (V1, V1_S)]):
    box(ax, 2 + i * 24, 43, 24, 6, f"shard{i}：rank{i} 读 ws 份归约", fc=cs, ec=c, fs=7.8)
arrow(ax, 50, 43, 50, 38)
box(ax, 20, 30, 60, 8, "归约结果原地写回所有 ws 个 rank 的 workspace 同位置",
    fc=PULL_S, ec=PULL, sub="→ 每个 workspace 都成为完整结果，all-gather 融合进写回", fs=8.8)
note(ax, 2, 27, "出口信号量带 release / acquire（写入 peer workspace 后 peer 要读）；graph 模式为真原地（out = in），eager 模式最后再 memcpy 回 output")

ax.text(0, 22, "multicast 档（2shot 的 NVLS 变体）", fontsize=9.8, color=MC,
        fontweight="bold", va="top")
box(ax, 2, 11, 32, 8, "multimem.ld_reduce.acc::f32", fc=MC_S, ec=MC,
    sub="一条指令：交换机归约 ws 份数据", fs=8.4)
arrow(ax, 34, 15, 40, 15, color=MC)
box(ax, 40, 11, 32, 8, "multimem.st", fc=MC_S, ec=MC,
    sub="一条指令：结果广播写回所有 rank", fs=8.4)
note(ax, 2, 8, "独立的 num_mc_blocks（sm100 ws=8 为 32 块 × 512 线程）——交换机归约吃不下满 grid")
note(ax, 2, 5, "bf16/fp16 的结果寄存器仍是 b32（.acc::f32 只提升累加精度），ptxas 拒绝 =f 目的寄存器")
note(ax, 2, 1.5, "PDL 贯穿三算法：入口 PDLWaitPrimary、出口 PDLTriggerSecondary；eager 的进出拷贝也用 PDL 版 memcpy_kernel",
     fs=8.4, color=PUSH)
save(fig, "custom_ar_fig4_v2_algos.png")


# --------------------------------------------------------------------------
# 图 5：v2 的 CUDA graph 指针表
# --------------------------------------------------------------------------
fig, ax = canvas(11.5, 5.4, "v2 的 graph 输入注册：capture 记账 → 退出批量交换 → replay 解引用")

ax.text(0, 94, "capture 期", fontsize=9.6, fontweight="bold", color=INK, va="top")
panel(ax, 2, 72, 44, 14, "每次 AR：记下 (input.data_ptr, nbytes)", [
    "把 graph_params[第 i 行] 的地址作为 pull_arg 传入 kernel",
    "行地址固定 → 烤进图里的参数永远合法",
], fc=PULL_S, ec=PULL, tc=PULL, fs=9.0)
arrow(ax, 46, 79, 52, 79)
panel(ax, 52, 72, 46, 14, "graph_params：[131072, ws] u64", [
    "第 i 行 = 第 i 个被捕获输入在全部 ws 个 rank 上的地址",
    "此刻内容仍为 0，待 capture 结束后回填",
], fs=9.0)

ax.text(0, 66, "capture 结束（capture() 上下文退出）", fontsize=9.6,
        fontweight="bold", color=INK, va="top")
box(ax, 2, 38, 96, 20, "", fc=PUSH_S, ec=PUSH)
for i, t in enumerate([
    "cudaMalloc 指针 → IPCManager.batch_get_handles → all_gather_object → batch_open_handles（批量，一轮交换）",
    "VMM 指针（expandable_segments）→ VmmGraphInputManager：按 allocation base 去重，fabric / posix-fd 重映射",
    "全部 peer 指针一次 copy_ 进表 → torch.cuda.synchronize()（表必须在任何 PDL 链 replay 前可见）",
]):
    note(ax, 3.6, 53 - i * 5.6, t, fs=8.6, color=INK)

ax.text(0, 32, "replay 期", fontsize=9.6, fontweight="bold", color=INK, va="top")
box(ax, 2, 12, 56, 12, "kernel 解引用自己那一行 → 直读各 rank 输入原址",
    fc=MC_S, ec=MC, align="left", fs=9.0,
    sub="零拷贝：eager 的进出 memcpy 消失；2shot 为真原地（out = in）")
panel(ax, 60, 12, 38, 12, "热路径守卫", [
    "_can_use_graph 同时判 _graph_mode_allowed 与",
    "is_current_stream_capturing()：warmup 不消耗表行",
], fs=9.0)
save(fig, "custom_ar_fig5_graph_table.png")


# --------------------------------------------------------------------------
# 图 6：sm100 ws=8 的尺寸分段与 16MB cap
# --------------------------------------------------------------------------
fig, ax = canvas(11.5, 4.6, "sm100（B200/B300）world_size=8 的尺寸 → 算法分段（对数轴示意）与 16 MB cap")

note(ax, 1, 82, "graph", fs=9.4, color=INK)
segs_g = [(10, 18, PUSH, PUSH_S, "1shot_push ≤512 KB"),
          (28, 28, PULL, PULL_S, "2shot_pull 512 KB–8 MB"),
          (56, 42, MC, MC_S, "2shot_pull · multicast 8 MB–128 MB")]
for x, w, c, cs, t in segs_g:
    box(ax, x, 76, w, 10, t, fc=cs, ec=c, fs=8.2)

note(ax, 1, 60, "eager", fs=9.4, color=INK)
segs_e = [(10, 21, PUSH, PUSH_S, "1shot_push ≤768 KB"),
          (31, 67, MC, MC_S, "2shot_pull · multicast 768 KB–128 MB（mc Range 覆盖全段）")]
for x, w, c, cs, t in segs_e:
    box(ax, x, 54, w, 10, t, fc=cs, ec=c, fs=8.2)

ax.plot([52, 52], [50, 90], color=PUSH, linewidth=1.4, linestyle=(0, (4, 2)), zorder=5)
note(ax, 53.5, 46, "默认 16 MB workspace cap：表值被 clip 到此，", fs=8.2, color=PUSH)
note(ax, 53.5, 42, "16 MB–128 MB 段实际不生效（除非调大 MAX_SIZE_KB）", fs=8.2, color=PUSH)

box(ax, 1, 4, 97, 30, "", fc="white", ec=HAIR)
ax.text(2.5, 29, "一个 TP8 bf16 模型（hidden 8192）的落点", fontsize=9.2,
        fontweight="bold", color=INK, va="center", zorder=3)
note(ax, 2.5, 21, "decode bs=1：16 KB → 1shot_push（graph 内同样是 push——push 只读本地输入，不需要指针表）",
     fs=8.6, color=INK)
note(ax, 2.5, 13,
     "prefill 8k token：[8192, 8192] ≈ 128 MB → 表上够得着 2shot · multicast，但 16 MB cap 先关门 → 回落 symm-mem / NCCL",
     fs=8.6, color=INK)
note(ax, 2.5, 7, "sm90（H100/H200）表形状类似但阈值小一个量级（ws=8 的 push 档仅 96 KB）", fs=8.2)
save(fig, "custom_ar_fig6_thresholds.png")
