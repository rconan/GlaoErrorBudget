import numpy as np
from gerpy import *

data = np.load("segKLmat.npz",allow_pickle=True)

for i in range(7):
    kl = KarhunenLoeve( modes=data['KL'][i].flatten(order='F').tolist(), n_mode=500, mask=data['mask'][i,:].flatten().tolist() )
    with open(f"M2S{i+1}.bin","wb") as f: 
      f.write(kl.bincode_serialize())
