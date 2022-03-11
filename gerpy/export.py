import numpy as np
from gerpy import *

data = np.load("../segKLmat.npz",allow_pickle=True)
id = 0
s1 = ASM__S1(value=Segment(modes=data['KL'][id].flatten(order='F').tolist(), n_mode=500, mask=data['mask'][id,:].flatten().tolist() ))
id = 1
s2 = ASM__S2(value=Segment(modes=data['KL'][id].flatten(order='F').tolist(), n_mode=500, mask=data['mask'][id,:].flatten().tolist() ))
id = 2
s3 = ASM__S3(value=Segment(modes=data['KL'][id].flatten(order='F').tolist(), n_mode=500, mask=data['mask'][id,:].flatten().tolist() ))
id = 3
s4 = ASM__S4(value=Segment(modes=data['KL'][id].flatten(order='F').tolist(), n_mode=500, mask=data['mask'][id,:].flatten().tolist() ))
id = 4
s5 = ASM__S5(value=Segment(modes=data['KL'][id].flatten(order='F').tolist(), n_mode=500, mask=data['mask'][id,:].flatten().tolist() ))
id = 5
s6 = ASM__S6(value=Segment(modes=data['KL'][id].flatten(order='F').tolist(), n_mode=500, mask=data['mask'][id,:].flatten().tolist() ))
id = 6
s7 = ASM__S7(value=Segment(modes=data['KL'][id].flatten(order='F').tolist(), n_mode=500, mask=data['mask'][id,:].flatten().tolist() ))

segs = [s1,s2,s3,s4,s5,s6,s7];
for i in range(7):
    with open(f"M2S{i+1}.bin","wb") as f: 
      f.write(segs[i].bincode_serialize())
