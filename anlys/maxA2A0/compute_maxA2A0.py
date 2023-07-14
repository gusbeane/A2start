import numpy as np
import arepo
import sys
from tqdm import tqdm
import glob
import os
import pickle
import h5py as h5
from numba import njit

from joblib import Parallel, delayed

def get_A2A0_profile(sn, Nbins=32):
    pos = sn.part2.pos.value
    R = np.linalg.norm(pos[:,:2], axis=1)

    key_sort = np.argsort(R)
    R = R[key_sort]
    pos = pos[key_sort]
    phi = np.arctan2(pos[:,1], pos[:,0])
    mass = np.full(sn.NumPart_Total[2], sn.MassTable[2].value)

    A2A0_list = []
    aveR_list = []

    key_split =np.array_split(np.arange(sn.NumPart_Total[2]), Nbins)
    for key in key_split:
        A2r = np.sum(mass[key] * np.cos(2*phi[key]))
        A2i = np.sum(mass[key] * np.sin(2*phi[key]))
        A0 = np.sum(mass[key])
    
        A2 = np.sqrt(A2r**2 + A2i**2)
    
        aveR = np.mean(R[key])
    
        aveR_list.append(aveR)
        A2A0_list.append(A2/A0)
    
    return np.array(aveR_list), np.array(A2A0_list)

def _runner(path, ic, name, snap, ptypes=[2]):
    sn = arepo.Snapshot(path + '/output/', snap, 
                        parttype=ptypes, 
                        fields=['Coordinates', 'Masses'],
                        combineFiles=True)
    
    R, A2A0 = get_A2A0_profile(sn)
    
    maxA2A0 = np.max(A2A0)
    maxR = R[np.argmax(A2A0)]
    
    Time = sn.Time.value

    # Package it all together
    output = (Time, R, A2A0, maxR, maxA2A0)
    
    return output

def run(path, ic, name, nsnap, nproc):

    out = Parallel(n_jobs=nproc) (delayed(_runner)(path, ic, name, i) for i in tqdm(range(nsnap)))

    Time     = np.array([out[i][0] for i in range(len(out))])
    R        = np.array([out[i][1] for i in range(len(out))])
    A2A0     = np.array([out[i][2] for i in range(len(out))])
    maxR     = np.array([out[i][3] for i in range(len(out))])
    maxA2A0  = np.array([out[i][4] for i in range(len(out))])

    out = {'Time'    : Time,
           'R'       : R,
           'A2A0'    : A2A0,
           'maxR'    : maxR,
           'maxA2A0' : maxA2A0}
    
    np.save('A2A0_'+name+'.npy', out)

if __name__ == '__main__':
    nproc = int(sys.argv[1])

    basepath = '../../'

    NbodyMND = 'Nbody-MND'

    pair_list = [(NbodyMND, 'lvl3'), # 0
                 (NbodyMND, 'lvl3-0.01'), # 1
                 (NbodyMND, 'lvl3-0.02'), # 2
                 (NbodyMND, 'lvl3-0.03'), # 3
                 (NbodyMND, 'lvl3-0.04'), # 4
                 (NbodyMND, 'lvl3-0.05'), # 5
                 ]


    name_list = [           p[0] + '-' + p[1] for p in pair_list]
    path_list = [basepath + 'runs/' + p[0] + '/' + p[1] for p in pair_list]
    ic_list   = [basepath + 'ics/' + p[0] + '/' + p[1] for p in pair_list]
    
    nsnap_list = [len(glob.glob(path+'/output/snapdir*/*.0.hdf5')) for path in path_list]
  
    i = int(sys.argv[2])
    path = path_list[i]
    name = name_list[i]
    nsnap = nsnap_list[i]
    ic = ic_list[i]

    out = run(path, ic, name, nsnap, nproc)
