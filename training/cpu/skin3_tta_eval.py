# usage: eval_tta.py PREFIX RUN_DIR  -> tune on val (skin2_tune), evaluate on test for TTA sets
import sys, json, shutil, numpy as np, pandas as pd, io, contextlib
from pathlib import Path
sys.path.insert(0,'/home/user/Vital360Ai-V2/training/cpu')
import skin2_tune
from scipy.special import softmax
from skin2_eval import metrics
from skin_decide import decide, TIER_IDX
from skin_classes import NAMES20 as NAMES
prefix, run = sys.argv[1], Path(sys.argv[2]); sets = {'none':[0],'flip2':[0,4],'flip4':[0,2,4,6],'rot4':[0,1,2,3],'all8':list(range(8))}
if len(sys.argv)>3: sets={k:v for k,v in sets.items() if k in sys.argv[3].split(',')}
Lv, Lt = np.load(prefix+'_val.npy'), np.load(prefix+'_test.npy')
sp = pd.read_csv(run/'split.csv'); te=np.where(sp.split=='test')[0]; t=sp.iloc[te].reset_index(drop=True)
y=t['cls'].map({n:i for i,n in enumerate(NAMES)}).to_numpy()
dark=np.load(run/'dark_test.npy') if (run/'dark_test.npy').exists() else np.load('/home/user/ml/runs/ft_v11/dark_test.npy')
tmp=Path(prefix+'_tmp'); tmp.mkdir(exist_ok=True)
for f in ('split.csv','dark_val.npy','dark_test.npy'):
    src = run/f if (run/f).exists() else Path('/home/user/ml/runs/ft_v11')/f
    shutil.copy(src, tmp/f)
res={}
for name,ks in sets.items():
    np.save(tmp/'val_logits_int8.npy', Lv[ks].mean(0))
    with contextlib.redirect_stdout(io.StringIO()): skin2_tune.main(tmp,tmp)
    tu=json.load(open(tmp/'tuning.json'))
    P=softmax(Lt[ks].mean(0)/tu['T'],axis=1)
    code=decide(P,tu['t_red'],tu['t_amber'],tu['thr'],True,tu['guard_min'],dark)
    r={}
    for s in ('overall','pad-ufes-20','scin'):
        m=np.ones(len(t),bool) if s=='overall' else (t.source==s).to_numpy()
        r[s]=metrics(P,code,y,m)
    nev=(y==NAMES.index('nevus'))&(t.source=='pad-ufes-20').to_numpy()
    r['tune']={k:tu[k] for k in ('T','t_red','t_amber','thr','guard_min')}; r['val']=tu['val']
    res[name]=r
    print(name, r['tune'])
    for s in ('overall','pad-ufes-20','scin'):
        m=r[s]; print(f"  {s:12s} tier {m['tier_accuracy']:.3f} sens {m['urgent_sensitivity']:.3f} spec {m['urgent_specificity']:.3f} hfa {m['healthy_false_alarm']:.3f} top1 {m['top1']:.3f} top3 {m['top3']:.3f} les1 {m['top1_lesion19']:.3f}")
    print('  val sens/spec', round(tu['val']['sens_red_shown'],3), round(tu['val']['spec_red'],3))
json.dump(res,open(prefix+'_res.json','w'),indent=1)
