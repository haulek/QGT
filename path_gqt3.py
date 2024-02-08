#!/usr/bin/env python
# @Copyright 2024 Kristjan Haule
Parallel = True
if Parallel :
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    msize = comm.Get_size()
    mrank = comm.Get_rank()
    master=0
else:
    msize = 1
    mrank = 0
    master = 0

import sys, os, re
from numpy import *
from numpy import linalg
import subprocess, shutil
import pickle 
import glob
from cmn import Ry2H,H2eV,Ry2eV,H2Ry,Br2Ang
import gwienfile as w2k

def PrintMC(A, file=sys.stdout):
    ni,nj = shape(A)
    for i in range(ni):
        for j in range(nj):
            print('%10.5f %10.5f '% (A[i,j].real,A[i,j].imag), end='', file=file)
        print(file=file)

def Get_case():
    struct_names = glob.glob('*.struct')
    if len(struct_names)==0:
        print('ERROR : Could not find a *.struct file present')
        sys.exit(0)
    else:
        case = struct_names[0].split('.')[0]
    return case
def Initialize(mode='w', Print=True, out_file='QGT.out'):
    if mrank==master:    
        fout = open(out_file, mode)
    else:
        fout = open(os.devnull,"w")
    case = Get_case()
    if Print and mrank==master: print('case=', case)
    return (case, fout)

def Find_nbs_nbei(ik, Ebnd, pairs_ik, nbs, nbe, fout):
    Ene = Ebnd[ik] # ks.Ebnd[ispin][ik]
    ibs,ibe = nbs,nbe
    while ibs>0:
        if abs(Ene[ibs]-Ene[ibs-1])<1e-6:
            ibs-=1
        else:
            break
    while ibe<len(Ene):
        if abs(Ene[ibe]-Ene[ibe-1])<1e-6:
            ibe+=1
        else:
            break
    
    corrected=False
    for il in range(len(pairs_ik)):
        jk = pairs_ik[il]
        Enj = Ebnd[jk] #ks.Ebnd[ispin][jk]
        while ibs>0:
            if abs(Enj[ibs]-Enj[ibs-1])<1e-6:
                ibs-=1
                corrected=True
            else:
                break
        while ibe<len(Enj):
            if abs(Enj[ibe]-Enj[ibe-1])<1e-6:
                ibe+=1
                corrected=True
            else:
                break
    return (ibs,ibe)

def Compute_QGT(nbs_nbe=None):
    # M[k,b]_{m,n}   = < psi_{m,k} | e^{-i b r} |psi_{n,k+b}>
    M_all = load('M_all.npy')
    klist = load('klist.npy')
    Ebnd = load('Ebnd.npy')
    with open('pairs.pkl', 'rb') as f:
        pairs = pickle.load(f)
        [nbs,nbe,tnbs,tnbe,latgen_Vol] = pickle.load(f)
        distances = pickle.load(f)
        idistance = pickle.load(f)
    nbs0,nbe0=nbs,nbe
    (case,fout) = Initialize('a')
    print('shape(klist)=', shape(klist))
    print('shape(M_all)=', shape(M_all))
    print('At reading: nbs,nbe,tnbs,tnbe=', nbs,nbe,tnbs,tnbe, file=fout)
    print('At reading: nbs,nbe,tnbs,tnbe=', nbs,nbe,tnbs,tnbe)
    #print('pairs=', pairs)
    #print('distances=', distances)
    #print('idistance=', idistance)
    strc = w2k.Struct(case, fout)
    latgen = w2k.Latgen(strc, fout)
    latgen.Symoper(strc, fout)
    # What W2k uses in stored klist. It is K2icartes@(i/N1,j/N2,k/N3)
    aaa = array([strc.a, strc.b, strc.c])
    if latgen.ortho or strc.lattice[1:3]=='CXZ':
        k2icartes = array((aaa/(2*pi)*latgen.br2).round(),dtype=int)
        k2cartes = 2*pi/aaa*identity(3) # == latgen.pia
        # When apply k2cartes . k2icartes . (i,j,k) we get
        # 2*pi/aaa[:] . BR2 . aaa[:]/(2*pi) (i,j,k) = 1/aaa[:] . BR2 . aaa[:]
    else:
        k2icartes = identity(3,dtype=int)
        k2cartes  = latgen.br2

    if nbs_nbe==None:
        nbs_nbe = [(i-nbs0,i+1-nbs0) for i in range(nbs,nbe)]
    elif nbs_nbe==(-1,-1):
        for iib,ib in enumerate(range(nbs,nbe)):
            Emin = min(Ebnd[:,ib])*H2eV
            Emax = max(Ebnd[:,ib])*H2eV
            print('band i=%3d (%3d) has range %12.6f : %12.6f eV' %(iib,ib,Emin,Emax))
        band_range = input('Which bands do you want to group into B?  is1:ie1,is2:ie2,... : ')
        print(band_range)
        mm = re.findall('(\d+)\s*:(\d+)', band_range)
        if mm is not None:
            #print(mm)
            nbs_nbe = [(int(m[0]),int(m[1])+1) for m in mm]
            print('nbs_nbe=', nbs_nbe)
        
    direction=['x','-x','y','-y','z','-z']
    QGM = []
    for iik,ik in enumerate(pairs):
        dst = zeros(3)
        for imu in range(3):
            # average over (|k+e_{imu}-k|^2 , |k-e_{imu}-k|^2)
            dst[imu]=0.5*(distances[idistance[iik,2*imu]]+distances[idistance[iik,2*imu+1]])
        tQGM=[]
        print('iik=', iik, 'ik=', ik, 'dst=%7.5f %7.5f %7.5f'%tuple(dst), klist[ik].tolist(), file=fout)
        for nbs,nbe in nbs_nbe:
            ibs,ibe = Find_nbs_nbei(ik, Ebnd, pairs[ik], nbs+nbs0, nbe+nbs0, fout)
            ibs-=nbs0
            ibe-=nbs0
            _ibs_ = ibs+nbs0-tnbs
            _ibe_ = ibe+nbs0-tnbs
            M_c = M_all[iik,:,_ibs_:_ibe_,_ibs_:_ibe_]
            print('  bands=',[ibs,nbs,nbe,ibe],[_ibs_,nbs+nbs0-tnbs,nbe+nbs0-tnbs,_ibe_],
                      ([i for i in range(nbs-ibs)],[i for i in range(nbs-ibs,nbe-ibs)],[i for i in range(nbe-ibs,ibe-ibs)]),
                      'M_mmn=', 'shape(M_all)=', shape(M_all[iik]), 'shape(M_c)=', shape(M_c), file=fout)
            #print(shape(M_c), file=fout)
            for imu in range(shape(M_all)[1]):
                print(' M_'+str(direction[imu]), file=fout)
                PrintMC(M_c[imu,:,:],file=fout)
            tR=zeros((nbe-nbs,3))
            ds=nbs-ibs
            # bands are arranged in [ibs,nbs,nbe,ibe]
            for imu in range(3):  # over x,y,z in lattice coordinates
                for i in range(nbs-ibs):
                    tR[0,imu] += 0.5*(1.0-sum(abs(M_c[  2*imu,:,i])**2)) # 1-\sum_m |<psi_{m,k}|psi_{i,k+e_{imu}}>|^2
                    tR[0,imu] += 0.5*(1.0-sum(abs(M_c[2*imu+1,:,i])**2)) # 1-\sum_m |<psi_{m,k}|psi_{i,k-e_{imu}}>|^2
                for i in range(nbs-ibs,nbe-ibs):
                    tR[i-ds,imu] += 0.5*(1.0-sum(abs(M_c[  2*imu,:,i])**2)) # 1-\sum_m |<psi_{m,k}|psi_{i,k+e_{imu}}>|^2
                    tR[i-ds,imu] += 0.5*(1.0-sum(abs(M_c[2*imu+1,:,i])**2)) # 1-\sum_m |<psi_{m,k}|psi_{i,k-e_{imu}}>|^2
                for i in range(nbe-ibs,ibe-ibs):
                    tR[-1,imu] += 0.5*(1.0-sum(abs(M_c[  2*imu,:,i])**2)) # 1-\sum_m |<psi_{m,k}|psi_{i,k+e_{imu}}>|^2
                    tR[-1,imu] += 0.5*(1.0-sum(abs(M_c[2*imu+1,:,i])**2)) # 1-\sum_m |<psi_{m,k}|psi_{i,k-e_{imu}}>|^2
            tR[0 ,:]  /= (nbs-ibs+1)
            tR[-1,:] /= (ibe-nbe+1)
            for imu in range(3):
                #tR[:,imu] *= (8*pi**2/latgen_Vol)/dst[imu]**2
                tR[:,imu] *= 2.0/dst[imu]**2
            #print('bands=', [ibs,nbs,nbe,ibe],file=fout)
            for i in range(0,nbe-nbs): #  over all the relevant bands, index i
                print('R[ik='+str(iik)+',i='+str(nbs+i)+']=', tR[i,:].tolist(),file=fout)
            tQGM.append(tR)
        QGM.append(tQGM)

    savetxt('nbs_nbe.dat', array(nbs_nbe,dtype=int))
    fq = open('QGT.dat','w')
    fe = open('eQGT.dat','w')
    dire=['x','y','z']
    print('# %2s %8s' % ('ik', '|d|'), end='',file=fq)
    for ib in range(len(nbs_nbe)):
        for i in range(nbs_nbe[ib][0],nbs_nbe[ib][1]):
            for imu in range(3):
                print('%12s '%('band['+str(i)+']:'+dire[imu],), end='', file=fq)
    print(file=fq)
                
    kp = klist[0]
    dst=0
    for iik,ik in enumerate(pairs):
        kn = array(klist[ik])
        dst += linalg.norm(k2cartes @ (kn-kp))
        print('%3d %8.5f' % (iik, dst), end='',file=fq)
        print('%10.5f' % (dst,), end='',file=fe)
        for ib,(nbs,nbe) in enumerate(nbs_nbe):
            for i in range(nbs,nbe):
                print('%12.5f'%(Ebnd[ik,i+nbs0]*H2eV,), end='', file=fe)
                for imu in range(3):
                    print('%12.4f ' % QGM[iik][ib][i-nbs,imu],  end='',file=fq)
        print(file=fe)
        print(file=fq)
        kp = kn
    print(file=fq)

if __name__ == '__main__':
    if mrank==master:
        Compute_QGT((-1,-1))
    # SrVO3: [(0,9),(9,12),(12,20)]
    # MnO:  [(0,3),(3,8)]
    # FeSe: [(0,6),(6,16),(16,23)]
    # LaRuSi: [(0,10),(10,17),(17,25)]
    # Si: [(0,4),(4,10)]
    # LSCO : [(0,16),(16,17)]
    # Bi2212 : [(0,44),(44,50),(51,55)]
    # CaRh2 : [(0,14),(14,18),(18,25)]
