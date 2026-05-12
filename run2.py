import numpy as np, time, sys
sys.path.insert(0,'/home/qec')
from surface_code import SurfaceCode
from classical_decoders import GreedyMWPM, UnionFind
from hybrid_decoder import HybridGABPNN

P = np.array([0.001,0.002,0.005,0.008,0.01,0.02,0.03,0.05,0.07,0.09,0.11,0.13,0.15])
DS = [3,5,7]; ETA = [1,10,100,1000]
N_TR,N_TE,EP,H,K = 5000,2000,30,128,3
RNG = np.random.default_rng(2024)

def ler(a,b): return float((a!=b).mean())

def train_hybrid(code,p_list,channel='depolarising',eta=1.0,n=1000):
    ts,tl=[],[]
    for p in p_list:
        s,l,_,_=code.generate_samples(n,p,channel=channel,eta=eta,rng=RNG)
        ts.append(s); tl.append(l)
    h=HybridGABPNN(code,hidden=H,bp_rounds=K,lr=8e-4,rng=RNG)
    h.train(np.vstack(ts),np.concatenate(tl),epochs=EP,batch_size=256,verbose=False)
    return h

# ── Symmetric ──────────────────────────────────────────────────────────────
SYM={dec:{d:[] for d in DS} for dec in ['mwpm','uf','hybrid']}
p_tr=[0.01,0.03,0.05,0.07,0.10]

for d in DS:
    print(f'd={d}',flush=True)
    code=SurfaceCode(d)
    M=GreedyMWPM(code); U=UnionFind(code)
    hyb=train_hybrid(code,p_tr,n=N_TR//5)
    for p in P:
        s,l,_,_=code.generate_samples(N_TE,p,rng=RNG)
        SYM['mwpm'][d].append(ler(M.decode_batch(s),l))
        SYM['uf'][d].append(ler(U.decode_batch(s),l))
        SYM['hybrid'][d].append(ler(hyb.decode_batch(s),l))
    print(f'  done',flush=True)

# ── Asymmetric (d=5) ───────────────────────────────────────────────────────
d=5; code=SurfaceCode(d)
M5=GreedyMWPM(code); U5=UnionFind(code)
ASYM={eta:{dec:[] for dec in ['mwpm','uf','hs','ha']} for eta in ETA}
hyb_s=train_hybrid(code,p_tr,n=800)  # symmetric trained

for eta in ETA:
    print(f'eta={eta}',flush=True)
    hyb_a=train_hybrid(code,p_tr,channel='asymmetric',eta=eta,n=800)
    for p in P:
        s,l,_,_=code.generate_samples(N_TE,p,channel='asymmetric',eta=eta,rng=RNG)
        ASYM[eta]['mwpm'].append(ler(M5.decode_batch(s),l))
        ASYM[eta]['uf'].append(ler(U5.decode_batch(s),l))
        ASYM[eta]['hs'].append(ler(hyb_s.decode_batch(s,eta=1.0),l))
        ASYM[eta]['ha'].append(ler(hyb_a.decode_batch(s,eta=eta),l))

# ── Timing ─────────────────────────────────────────────────────────────────
TIM={}
for d in DS:
    code=SurfaceCode(d); rng2=np.random.default_rng(99)
    s,l,_,_=code.generate_samples(300,0.05,rng=rng2)
    t0=time.time(); GreedyMWPM(code).decode_batch(s); TIM[f'm{d}']=(time.time()-t0)/300*1e6
    t0=time.time(); UnionFind(code).decode_batch(s);  TIM[f'u{d}']=(time.time()-t0)/300*1e6
    h2=HybridGABPNN(code,hidden=H,bp_rounds=K,rng=rng2)
    h2.train(s,l,epochs=5,batch_size=64); t0=time.time()
    h2.decode_batch(s); TIM[f'h{d}']=(time.time()-t0)/300*1e6

save_d = dict(p_values=P, distances=DS, eta_values=ETA,
    **{f'sym_{dec}_{d}':SYM[dec][d] for dec in ['mwpm','uf','hybrid'] for d in DS},
    **{f'a{eta}_{dec}':ASYM[eta][dec] for eta in ETA for dec in ['mwpm','uf','hs','ha']},
    **TIM)
np.savez('/home/qec/results.npz',**save_d)
print('SAVED',flush=True)
