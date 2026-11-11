import sys, numpy as np, onnxruntime as ort, pandas as pd, time
sys.path.insert(0,'/home/user/Vital360Ai-V2/training/cpu')
from skin_common import to_tensor
onnx, out = sys.argv[1], sys.argv[2]
E=np.load('/home/user/ml/cache2/eval224.npy',mmap_mode='r'); sp=pd.read_csv('/home/user/ml/runs/ft_v11/split.csv')
s=ort.InferenceSession(onnx,providers=['CPUExecutionProvider'])
def tf(x,k): # x CHW; k in 0..7: rot90 k%4, flip if k>=4
    x=np.rot90(x,k%4,axes=(1,2)); 
    if k>=4: x=x[:,:,::-1]
    return np.ascontiguousarray(x)
for split in ('val','test'):
    ii=np.where(sp.split==split)[0]; X=np.stack([to_tensor(E[i]) for i in ii]); t0=time.time()
    L=np.zeros((8,len(ii),20),np.float32)
    for k in range(8):
        xs=np.stack([tf(x,k) for x in X])
        L[k]=np.concatenate([s.run(None,{'input':xs[i:i+1]})[0] for i in range(len(xs))])
        print(split,k,time.time()-t0,flush=True)
    np.save(f'{out}_{split}.npy',L)
