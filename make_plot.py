import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BG   = '#000000'
FG   = '#ffffff'
GREY = '#666666'

CATS    = ['Coding', 'Data & ML', 'Games', 'Math & Reasoning', 'Search & Tool Use']
CAT_CLR = ['#00e5ff', '#69ff47', '#ff4081', '#ffb300', '#e040fb']

RTYPES  = ['Binary', 'Binary Weighted', 'Correctness + Bonus', 'Perf-Tiers',
           'Multi-Component', 'LLM-Judge', 'LLM-Judge + Penalty',
           'Rubric / Deduction', 'Custom Scorer']
RT_CLR  = ['#00e5ff', '#40c4ff', '#69ff47', '#c6ff00',
           '#ffb300', '#ff4081', '#e040fb', '#ff6e40', '#888888']

ENVS = [
    ('kernelbench',       0, 4, 'kernelbench'),
    ('nanogpt-speedrun',  0, 2, 'nanogpt-speedrun'),
    ('math-python',       0, 0, 'math-python'),
    ('mini-swe',          0, 0, 'mini-swe'),
    ('oolong-rlm',        0, 5, 'oolong-rlm'),
    ('pmpp',              1, 1, 'pmpp'),
    ('agentclinic-nejm',  1, 0, 'agentclinic-nejm'),
    ('agentclinic-ext',   1, 0, 'agentclinic-ext'),
    ('agency-bench',      1, 7, 'agency-bench'),
    ('med-agent-bench',   1, 0, 'med-agent-bench'),
    ('wordle',            2, 4, 'wordle'),
    ('hud-text-2048',     2, 4, 'hud-text-2048'),
    ('hud-browser-2048',  2, 4, 'hud-browser-2048'),
    ('lights-out',        2, 4, 'lights-out'),
    ('balrog-prime',      2, 4, 'balrog-prime'),
    ('reasoning-core',    3, 8, 'reasoning-core'),
    ('GoodSirMath8k',     3, 4, 'GoodSirMath8k'),
    ('minif2f',           3, 0, 'minif2f'),
    ('frontierscience',   3, 5, 'frontierscience'),
    ('math-python-2',     3, 2, 'math-python*'),
    ('acebench',          4, 4, 'acebench'),
    ('wiki-search',       4, 5, 'wiki-search'),
    ('deepdive',          4, 6, 'deepdive'),
    ('pi-wiki-search',    4, 5, 'pi/wiki-search'),
    ('pi-deepdive',       4, 2, 'pi/deepdive'),
]

# ── layout ─────────────────────────────────────────────────────────────────────
# Figure is 26 wide: bipartite flow in [0..21], inset panel in [21.5..26]
FW, FH  = 26, 13
FLOW_W  = 21.0    # bipartite flow occupies x: 0 → FLOW_W
INS_X   = 21.6    # inset panel starts here

FONT_LG = 13
FONT_MD = 9
FONT_SM = 8

BOX_W   = 3.1
BOX_H   = 0.70
BOX_LW  = 1.5

CAT_X0  = 0.20
CAT_CX  = CAT_X0 + BOX_W / 2
RT_X0   = FLOW_W - 0.20 - BOX_W
RT_CX   = RT_X0 + BOX_W / 2
ENVX    = (CAT_X0 + BOX_W + RT_X0) / 2

TOP_Y   = 12.0
BOT_Y   = 1.55
cat_ys  = np.linspace(TOP_Y - 0.4, BOT_Y + 0.3, len(CATS))
rt_ys   = np.linspace(TOP_Y - 0.2, BOT_Y - 0.1, len(RTYPES))

fig = plt.figure(figsize=(FW, FH), facecolor=BG)
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_facecolor(BG); ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis('off')

# thin divider between flow and inset panel
ax.plot([INS_X - 0.15, INS_X - 0.15], [0.3, FH - 0.3],
        color='#222222', lw=0.8, zorder=0)

# ── title (centered over flow area only) ─────────────────────────────────────
ax.text(FLOW_W / 2, 12.68,
        'Prime Intellect Environments  —  Reward Function Taxonomy',
        ha='center', va='center', color=FG, fontsize=FONT_LG,
        fontweight='bold', fontfamily='monospace')
ax.text(FLOW_W / 2, 12.34,
        'Top-5 per category  x  5 categories  |  reward type classified from source code',
        ha='center', va='center', color=GREY, fontsize=FONT_SM,
        fontfamily='monospace')

# ── category boxes ────────────────────────────────────────────────────────────
for i, (cat, cy) in enumerate(zip(CATS, cat_ys)):
    col = CAT_CLR[i]
    ax.add_patch(mpatches.FancyBboxPatch(
        (CAT_X0, cy - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.07", lw=BOX_LW,
        edgecolor=col, facecolor='#0c0c0c'))
    ax.text(CAT_CX, cy, cat, ha='center', va='center',
            color=col, fontsize=FONT_MD, fontweight='bold',
            fontfamily='monospace')

# ── reward-type boxes ─────────────────────────────────────────────────────────
for ri, (rt, ry) in enumerate(zip(RTYPES, rt_ys)):
    col = RT_CLR[ri]
    ax.add_patch(mpatches.FancyBboxPatch(
        (RT_X0, ry - BOX_H / 2), BOX_W, BOX_H,
        boxstyle="round,pad=0.07", lw=BOX_LW,
        edgecolor=col, facecolor='#0c0c0c'))
    ax.text(RT_CX, ry, rt, ha='center', va='center',
            color=col, fontsize=FONT_MD, fontweight='bold',
            fontfamily='monospace')

# ── env dots + labels ─────────────────────────────────────────────────────────
cat_members = {i: [] for i in range(len(CATS))}
for rec in ENVS:
    cat_members[rec[1]].append(rec)

env_pos = {}
for ci, members in cat_members.items():
    cy = cat_ys[ci]; n = len(members)
    ys = np.linspace(cy + 0.40*(n-1)/2, cy - 0.40*(n-1)/2, n)
    for rec, ey in zip(members, ys):
        name, _, ri, lbl = rec
        env_pos[name] = (ENVX, ey, ri, lbl)
        col = RT_CLR[ri]
        ax.plot(ENVX, ey, 'o', color=col, markersize=5.5, zorder=5)
        ax.text(ENVX + 0.17, ey, lbl, ha='left', va='center',
                color=FG, fontsize=FONT_SM, fontfamily='monospace', zorder=5)

# ── connection lines ──────────────────────────────────────────────────────────
for ci, members in cat_members.items():
    cy = cat_ys[ci]; cat_col = CAT_CLR[ci]
    for rec in members:
        name = rec[0]
        ex, ey, ri, _ = env_pos[name]
        rt_col = RT_CLR[ri]; ry = rt_ys[ri]
        ax.plot([CAT_X0 + BOX_W, ex - 0.12], [cy, ey],
                color=cat_col, lw=0.6, alpha=0.25, zorder=1)
        ax.plot([ex + 0.12, RT_X0], [ey, ry],
                color=rt_col, lw=0.8, alpha=0.50, zorder=1)

# ── 20Q compact highlight row ─────────────────────────────────────────────────
Q20_Y  = 0.78
Q20_X0 = CAT_X0 + BOX_W + 0.35
Q20_W  = RT_X0 - Q20_X0 - 0.35
Q20_H  = BOX_H

ax.add_patch(mpatches.FancyBboxPatch(
    (Q20_X0, Q20_Y - Q20_H / 2), Q20_W, Q20_H,
    boxstyle="round,pad=0.07", lw=2.0,
    edgecolor='#69ff47', facecolor='#011a01', zorder=8))
ax.text(Q20_X0 + 0.26, Q20_Y, '★ twenty-questions',
        ha='left', va='center', color='#69ff47',
        fontsize=FONT_MD, fontweight='bold', fontfamily='monospace', zorder=9)
ax.text(Q20_X0 + Q20_W - 0.22, Q20_Y,
        'Correctness + Efficiency Bonus  |  sparse  |  EIG = diagnostic only',
        ha='right', va='center', color='#7acc7a',
        fontsize=FONT_SM - 0.5, fontfamily='monospace', zorder=9)

# ── inset axes: distribution bar chart ───────────────────────────────────────
# Lives in its own clear column to the right of the divider
# figure coords: left = INS_X/FW, bottom = 0.08, width, height
ins_l = INS_X / FW + 0.005
ins_b = 0.08
ins_w = 1.0 - ins_l - 0.015
ins_h = 0.86

ax_ins = fig.add_axes([ins_l, ins_b, ins_w, ins_h])
ax_ins.set_facecolor('#080808')
for spine in ax_ins.spines.values():
    spine.set_edgecolor('#333333'); spine.set_linewidth(0.8)
ax_ins.tick_params(colors=GREY, labelsize=7, length=2, width=0.6)

counts  = {rt: 0 for rt in RTYPES}
for rec in ENVS:
    counts[RTYPES[rec[2]]] += 1

bar_vals = [counts[rt] for rt in RTYPES]
ax_ins.barh(range(len(RTYPES)), bar_vals, color=RT_CLR, alpha=0.80, height=0.60)
ax_ins.set_yticks(range(len(RTYPES)))
ax_ins.set_yticklabels(RTYPES, fontsize=7, fontfamily='monospace', color=GREY)
ax_ins.set_xticks(range(0, max(bar_vals) + 2, 2))
ax_ins.xaxis.set_tick_params(labelsize=7)
ax_ins.set_xlabel('# envs', fontsize=7, color=GREY, fontfamily='monospace', labelpad=3)
ax_ins.set_title('Distribution', fontsize=8, color=FG,
                 fontfamily='monospace', pad=6, fontweight='bold')
ax_ins.set_xlim(0, max(bar_vals) + 1.5)
ax_ins.set_ylim(-0.6, len(RTYPES) - 0.4)
ax_ins.invert_yaxis()
ax_ins.tick_params(axis='x', colors=GREY)
ax_ins.tick_params(axis='y', colors=GREY)
ax_ins.xaxis.label.set_color(GREY)

for i, v in enumerate(bar_vals):
    if v > 0:
        ax_ins.text(v + 0.08, i, str(v), va='center',
                    color=RT_CLR[i], fontsize=7.5, fontweight='bold',
                    fontfamily='monospace')

plt.savefig('E:/prime-envs/reward_taxonomy.png',
            dpi=150, bbox_inches='tight', facecolor=BG, pad_inches=0.12)
print("saved")
