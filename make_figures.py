"""figures from simulation results"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.gridspec import GridSpec

r   = np.load('/home/qec/results.npz', allow_pickle=True)
P   = r['p_values']
OUT = '/home/qec/'

COLORS = {'mwpm':'#E63946','uf':'#F4A261','hybrid':'#2A9D8F'}
LABELS = {'mwpm':'MWPM','uf':'Union-Find','hybrid':'GABPNN (Ours)'}
MARK   = {'mwpm':'o','uf':'s','hybrid':'D'}
DSTYLE = {3:'-',5:'--',7:'-.'}

def _style():
    plt.rcParams.update({
        'font.family':'serif','font.size':11,
        'axes.spines.top':False,'axes.spines.right':False,
        'axes.grid':True,'grid.alpha':0.35,'grid.linestyle':':',
        'figure.dpi':150,'savefig.dpi':180,
        'legend.framealpha':0.9,'legend.edgecolor':'#cccccc'
    })

_style()

# ── Figure 1: P_L vs p (symmetric), 3 distances, 3 decoders ─────────────────
fig, axes = plt.subplots(1,3,figsize=(14,4.5),sharey=False)
for ax, d in zip(axes, [3,5,7]):
    for dec in ['mwpm','uf','hybrid']:
        y = np.array(r[f'sym_{dec}_{d}'])
        ax.semilogy(P, y, marker=MARK[dec], color=COLORS[dec],
                    label=LABELS[dec], lw=2, ms=6,
                    markerfacecolor='white', markeredgewidth=1.5)
    ax.set_title(f'Distance d = {d}', fontweight='bold')
    ax.set_xlabel('Physical Error Rate  p')
    ax.set_ylabel('Logical Error Rate  $P_L$')
    ax.set_xlim(P[0]-0.001, P[-1]+0.005)
    ax.yaxis.set_major_formatter(ticker.LogFormatter())
    if d == 3:
        ax.legend(fontsize=9)
    # threshold region shade
    ax.axvspan(0.09,0.11,alpha=0.08,color='purple',label='~threshold')

plt.suptitle('Figure 1: Logical Error Rate vs Physical Error Rate — Symmetric Depolarising Channel',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT+'fig1_sym_ler.pdf', bbox_inches='tight')
plt.savefig(OUT+'fig1_sym_ler.png', bbox_inches='tight')
plt.close()
print('Fig 1 done')

# ── Figure 2: All distances on one axis per decoder (distance scaling) ────────
fig, axes = plt.subplots(1,3,figsize=(14,4.5))
dec_labels = ['mwpm','uf','hybrid']
dist_colors = {3:'#1d3557',5:'#457b9d',7:'#a8dadc'}
for ax, dec in zip(axes, dec_labels):
    for d in [3,5,7]:
        y = np.array(r[f'sym_{dec}_{d}'])
        ax.semilogy(P, y, marker=MARK[dec], color=dist_colors[d],
                    label=f'd={d}', lw=2, ms=6,
                    markerfacecolor='white', markeredgewidth=1.5)
    ax.set_title(LABELS[dec], fontweight='bold')
    ax.set_xlabel('Physical Error Rate  p')
    ax.set_ylabel('Logical Error Rate  $P_L$')
    ax.legend(fontsize=9)
plt.suptitle('Figure 2: Distance Scaling — Logical Error Suppression Below Threshold',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT+'fig2_distance_scaling.pdf', bbox_inches='tight')
plt.savefig(OUT+'fig2_distance_scaling.png', bbox_inches='tight')
plt.close()
print('Fig 2 done')

# ── Figure 3: Hybrid gain over MWPM (relative improvement) ───────────────────
fig, axes = plt.subplots(1,3,figsize=(14,4.2))
for ax, d in zip(axes,[3,5,7]):
    mwpm_y   = np.array(r[f'sym_mwpm_{d}'])
    hyb_y    = np.array(r[f'sym_hybrid_{d}'])
    gain_pct = (mwpm_y - hyb_y) / (mwpm_y + 1e-12) * 100
    uf_y     = np.array(r[f'sym_uf_{d}'])
    gain_uf  = (uf_y  - hyb_y) / (uf_y  + 1e-12) * 100
    ax.bar(P, gain_pct, width=0.006, color=COLORS['mwpm'],
           alpha=0.7, label='vs MWPM')
    ax.bar(P, gain_uf,  width=0.003, color=COLORS['uf'],
           alpha=0.7, label='vs Union-Find', align='edge')
    ax.axhline(0, color='k', lw=0.8)
    ax.set_title(f'd = {d}', fontweight='bold')
    ax.set_xlabel('p'); ax.set_ylabel('Improvement (%)')
    if d==3: ax.legend(fontsize=9)
plt.suptitle('Figure 3: GABPNN Relative Improvement over Classical Decoders (%)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT+'fig3_improvement.pdf', bbox_inches='tight')
plt.savefig(OUT+'fig3_improvement.png', bbox_inches='tight')
plt.close()
print('Fig 3 done')

# ── Figure 4: Asymmetric channel at p=0.05 ──────────────
ETA_VALS = [1,10,100,1000]
p_idx    = list(P).index(0.05) if 0.05 in P else 5

fig, axes = plt.subplots(1,2,figsize=(12,4.5))
dec_keys = ['mwpm','uf','hs','ha']
dec_lb   = ['MWPM','Union-Find','GABPNN (sym. train)','GABPNN (asym. train)']
bar_col  = [COLORS['mwpm'], COLORS['uf'], '#6a4c93','#2A9D8F']

x = np.arange(len(ETA_VALS)); w = 0.18
for ax_i, ax in enumerate(axes):
    title_p = P[p_idx] if ax_i==0 else P[3]
    pi = p_idx if ax_i==0 else 3
    for j,(k,lb,c) in enumerate(zip(dec_keys,dec_lb,bar_col)):
        vals = [r[f'a{eta}_{k}'][pi] for eta in ETA_VALS]
        ax.bar(x + j*w, vals, w, label=lb, color=c, alpha=0.85, edgecolor='white')
    ax.set_xticks(x + 1.5*w); ax.set_xticklabels([f'η={e}' for e in ETA_VALS])
    ax.set_title(f'p = {title_p:.3f}', fontweight='bold')
    ax.set_ylabel('Logical Error Rate $P_L$')
    ax.legend(fontsize=8)

plt.suptitle('Figure 4: Logical Error Rate under Asymmetric (Z-biased) Channel, d=5',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT+'fig4_asymmetric.pdf', bbox_inches='tight')
plt.savefig(OUT+'fig4_asymmetric.png', bbox_inches='tight')
plt.close()
print('Fig 4 done')

# ── Figure 5: Asymmetric sweep (P_L vs p) for eta=100 ────────────────────────
fig, axes = plt.subplots(1,2,figsize=(12,4.5))
eta_plot=[10,100]
dec_keys2=['mwpm','uf','hs','ha']
dec_lb2=['MWPM','Union-Find','GABPNN (sym.)','GABPNN (asym.)']
dec_col2=[COLORS['mwpm'],COLORS['uf'],'#6a4c93','#2A9D8F']
dec_mk=['o','s','^','D']
for ax,eta in zip(axes,eta_plot):
    for k,lb,c,m in zip(dec_keys2,dec_lb2,dec_col2,dec_mk):
        y=np.array(r[f'a{eta}_{k}'])
        ax.semilogy(P,y,marker=m,color=c,label=lb,lw=2,ms=6,
                    markerfacecolor='white',markeredgewidth=1.5)
    ax.set_title(f'η = {eta}', fontweight='bold')
    ax.set_xlabel('p'); ax.set_ylabel('$P_L$')
    ax.legend(fontsize=9)
plt.suptitle('Figure 5: P_L vs p under Asymmetric Channel (d=5)',
             fontsize=12, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(OUT+'fig5_asym_sweep.pdf', bbox_inches='tight')
plt.savefig(OUT+'fig5_asym_sweep.png', bbox_inches='tight')
plt.close()
print('Fig 5 done')

# ── Figure 6: Decoding time (µs/sample) ──────────────────────────────────────
fig, ax = plt.subplots(figsize=(7,4))
bar_w = 0.25
x = np.arange(3)
ax.bar(x-bar_w, [r['m3'],r['m5'],r['m7']], bar_w, label='MWPM',       color=COLORS['mwpm'],   alpha=0.85)
ax.bar(x,       [r['u3'],r['u5'],r['u7']], bar_w, label='Union-Find', color=COLORS['uf'],     alpha=0.85)
ax.bar(x+bar_w, [r['h3'],r['h5'],r['h7']], bar_w, label='GABPNN',     color=COLORS['hybrid'], alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(['d=3','d=5','d=7'])
ax.set_ylabel('Decoding Time (µs / sample)')
ax.set_title('Figure 6: Decoder Latency Comparison', fontweight='bold')
ax.legend()
plt.tight_layout()
plt.savefig(OUT+'fig6_timing.pdf', bbox_inches='tight')
plt.savefig(OUT+'fig6_timing.png', bbox_inches='tight')
plt.close()
print('Fig 6 done')

print('\nAll figures saved.')
