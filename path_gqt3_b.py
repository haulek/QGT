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

def Compute_QGT(Delta, nbs_nbe=None):
    
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

    datE = loadtxt(case+'.eig').T
    for n_bands in range(1000):
        if datE[0][n_bands]!=(n_bands+1): break
    print('n=', n_bands)
    nkp = int(datE[1][::n_bands][-1])
    Eband = reshape(datE[2],(nkp,n_bands))

    print('shape(klist)=', shape(klist))
    print('shape(M_all)=', shape(M_all))
    print('At reading: nbs,nbe,tnbs,tnbe=', nbs,nbe,tnbs,tnbe, file=fout)
    print('At reading: nbs,nbe,tnbs,tnbe=', nbs,nbe,tnbs,tnbe)
    #print('pairs=', pairs)
    #print('distances=', distances)
    #print('idistance=', idistance)
    strc = w2k.Struct(case, fout)
    latgen = w2k.Latgen(strc, fout)
    # latgen.Symoper(strc, fout) I think symope is called in Latgen initialization
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
    print('nbs_nbe=', nbs_nbe)
    direction=['x','-x','y','-y','z','-z']
    QGM = []
    QGR = []
    GeD = []
    QSM = []
    CoD=[]
    for iik,ik in enumerate(pairs):
        dst = zeros(3)
        for imu in range(3):
            # average over (|k+e_{imu}-k|^2 , |k-e_{imu}-k|^2)
            dst[imu]=0.5*(distances[idistance[iik,2*imu]]+distances[idistance[iik,2*imu+1]])
        tQGM=[]
        tQRM=[]
        tGeD=[]
        tQSM=[]
        tCD=[]
        print('iik=', iik, 'ik=', ik, 'dst=%7.5f %7.5f %7.5f'%tuple(dst), klist[ik].tolist(), file=fout)
        for nbs,nbe in nbs_nbe:
            # nbs,nbe are first&last band needed for this unit B, at this particular k-point
            # tnbs,tnbe are first&last band over all k-points for all needed Bs.
            # ibs,ibe are first&last point but taking into account degeneracies
            ibs,ibe = Find_nbs_nbei(ik, Ebnd, pairs[ik], nbs+nbs0, nbe+nbs0, fout)
            ibs-=nbs0
            ibe-=nbs0
            if ibs<0: ibs=0
            if (ibe > shape(M_all)[2] + tnbs-nbs0): ibe=nbe
            
            _ibs_ = ibs+nbs0-tnbs
            _ibe_ = ibe+nbs0-tnbs

            M_c = M_all[iik,:,_ibs_:_ibe_,_ibs_:_ibe_] # [k,{x,y,z},bands,bands]
            complement = list(range(_ibs_))+list(range(_ibe_,shape(M_all)[2]))
            M_a = M_all[iik,:,complement,_ibs_:_ibe_] # <psi_{n,k}|psi_{m,k+e_{imu}>
            M_b = M_all[iik,:,_ibs_:_ibe_,complement] # <psi_{m,k}|psi_{n,k+e_{imu}>
            print('nbs=', nbs, 'nbe=', nbe, 'ibs=', ibs, 'ibe=', ibe, '_ibs_=', _ibs_, '_ibe_=', _ibe_, 'tnbs=', tnbs, 'tnbe=', tnbe, file=fout)
            print(f"three loops=0:{nbs-ibs}, {nbs-ibs}:{nbe-ibs}, {nbe-ibs}:{ibe-ibs}", file=fout)
            print('shape(M_a)=', shape(M_a), 'shape(M_b)=', shape(M_b), 'shape(M_all)=', shape(M_all), 'shape(M_c)=', shape(M_c), file=fout)
            print('E_in=', Ebnd[ik,ibs+nbs0:ibe+nbs0], file=fout)
            complement2 = list(arange(ibs)+nbs0)+list(arange(ibe,shape(M_all)[2])+nbs0) # here is probably not shape(M_all)
            #print('ns+nbs0=', nbs+nbs0, 'nbe+nbs0=', nbe+nbs0, 'ibs+nbs0=', ibs+nbs0, 'ibe+nbs0=', ibs+nbs0)
            #print('complement=', complement, shape(complement), file=fout)
            #print('complement2=', complement2, shape(complement2), file=fout)
            E_in = Ebnd[ik,nbs+nbs0:nbe+nbs0]*H2eV
            #E_in = Ebnd[ik,ibs+nbs0:ibe+nbs0]*H2eV
            E_out = Ebnd[ik,complement2]*H2eV

            dE_imu = zeros((nbe-nbs,3)) # cartesian derivative of energy de/dk_i = (e(k+dk_i)-e(k-dk_i))/(2*dk_i)
            for imu in range(3):
                # note that E[ik:ik+7] = [ek(k),ek(k+ex),ek(k-ex),ek(k+ey),ek(k-ey),ek(k+ez),ek(k-ez)]
                E_p = Ebnd[ik+1+2*imu,nbs+nbs0:nbe+nbs0]*H2eV
                E_m = Ebnd[ik+2+2*imu,nbs+nbs0:nbe+nbs0]*H2eV
                dE_imu[:,imu] = (E_p-E_m)/(2*dst[imu])
                
            #print('E_in=', E_in)
            #print('E_out=', E_out)
            
            # M_all[k,idir,m,n] should be <u_m(k)|u_n(k+e_{idir}*small)>
            print('  bands=',[ibs,nbs,nbe,ibe],[_ibs_,nbs+nbs0-tnbs,nbe+nbs0-tnbs,_ibe_],
                      ([i for i in range(nbs-ibs)],[i for i in range(nbs-ibs,nbe-ibs)],[i for i in range(nbe-ibs,ibe-ibs)]),
                      'M_mmn=', 'shape(M_all)=', shape(M_all[iik]), 'shape(M_c)=', shape(M_c),
                      'shape(M_a)=', shape(M_a), 'shape(M_b)=', shape(M_b), file=fout)
            #print(shape(M_c), file=fout)
            for imu in range(shape(M_all)[1]):
                print(' M_'+str(direction[imu]), file=fout)
                PrintMC(M_c[imu,:,:],file=fout)
            tR=zeros((nbe-nbs,3))
            tQ=zeros((nbe-nbs,3))
            tG=zeros((nbe-nbs,3))
            tS=zeros((nbe-nbs,3))
            tC=zeros((nbe-nbs,3))
            ds=nbs-ibs

            #print('ik=', ik, 'shape(M_a)=', shape(M_a), 'shape(E_out)=', shape(E_out), 'shape(E_in)=', shape(E_in))
            
            # bands are arranged in the following order [ibs,nbs,nbe,ibe]
            # Note that all bands between [ibs,nbs] are degenerate with the band at nbs, i.e., at the bottom.
            # Note that all bands between [nbe,ibe] are degenerate with the band at nbe, i.e., at the top.
            # The bands between [nbs,nbe] constituate our group B, and should be always computed.
            for imu in range(3):  # over x,y,z in lattice coordinates
                for i in range(nbs-ibs): # This is for those bands that might be degenerate at the bottom
                    tR[0,imu] += 0.25*(2.0-sum(abs(M_c[  2*imu,:,i])**2)-sum(abs(M_c[  2*imu,i,:])**2))# 2-\sum_m(|<psi_{m,k}|psi_{i,k+e_{imu}}>|^2+|<psi_{i,k}|psi_{m,k+e_{imu}}>|^2)
                    tR[0,imu] += 0.25*(2.0-sum(abs(M_c[2*imu+1,:,i])**2)-sum(abs(M_c[2*imu+1,i,:])**2))# 2-\sum_m(|<psi_{m,k}|psi_{i,k-e_{imu}}>|^2+|<psi_{i,k}|psi_{m,k-e_{imu}}>|^2)
                for i in range(nbs-ibs,nbe-ibs): # if no degeneracies, this is the only loop
                    tR[i-ds,imu] += 0.25*(2.0-sum(abs(M_c[  2*imu,:,i])**2)-sum(abs(M_c[  2*imu,i,:])**2)) # 2-\sum_m(|<psi_{m,k}|psi_{i,k+e_{imu}}>|^2+|<psi_{i,k}|psi_{m,k+e_{imu}}>|^2)
                    tR[i-ds,imu] += 0.25*(2.0-sum(abs(M_c[2*imu+1,:,i])**2)-sum(abs(M_c[2*imu+1,i,:])**2)) # 2-\sum_m(|<psi_{m,k}|psi_{i,k-e_{imu}}>|^2+|<psi_{i,k}|psi_{m,k-e_{imu}}>|^2)
                for i in range(nbe-ibs,ibe-ibs): # I think this is for those bands that might be degenerate at the top
                    tR[-1,imu] += 0.25*(2.0-sum(abs(M_c[  2*imu,:,i])**2)-sum(abs(M_c[  2*imu,i,:])**2)) # 2-\sum_m(|<psi_{m,k}|psi_{i,k+e_{imu}}>|^2+|<psi_{i,k}|psi_{m,k+e_{imu}}>|^2)
                    tR[-1,imu] += 0.25*(2.0-sum(abs(M_c[2*imu+1,:,i])**2)-sum(abs(M_c[2*imu+1,i,:])**2)) # 2-\sum_m(|<psi_{m,k}|psi_{i,k-e_{imu}}>|^2+|<psi_{i,k}|psi_{m,k-e_{imu}}>|^2)
                    
                for i in range(nbs-ibs):
                    tQ[0,imu] += 0.25*(sum(abs(M_a[:,2*imu,i])**2)+sum(abs(M_a[:,2*imu+1,i])**2))  # sum_{n\in\complement}(|<psi_{n,k}|psi_{i,k+e_imu}>|^2+|<psi_{n,k}|psi_{i,k-e_imu}>|^2)
                    tQ[0,imu] += 0.25*(sum(abs(M_b[:,2*imu,i])**2)+sum(abs(M_b[:,2*imu+1,i])**2))  # sum_{n\in\complement}(|<psi_{i,k}|psi_{n,k+e_imu}>|^2+|<psi_{i,k}|psi_{n,k-e_imu}>|^2)
                    rE0 = (E_out[:]-E_in[0])/(E_out[:]+E_in[0])
                    #print('shape(M_a)=', shape(M_a)[0], 'shape(E_out)=', shape(E_out))
                    rE = rE0 * (1/sqrt(E_in[0]**2+Delta**2)-1/sqrt(E_out[:]**2+Delta**2))
                    tG[0,imu] += 0.25*(sum(abs(M_a[:,2*imu,i])**2*rE)+sum(abs(M_a[:,2*imu+1,i])**2*rE))  # sum_{n\in\complement}(|<psi_{n,k}|psi_{i,k+e_imu}>|^2+|<psi_{n,k}|psi_{i,k-e_imu}>|^2)
                    tG[0,imu] += 0.25*(sum(abs(M_b[:,2*imu,i])**2*rE)+sum(abs(M_b[:,2*imu+1,i])**2*rE))  # sum_{n\in\complement}(|<psi_{i,k}|psi_{n,k+e_imu}>|^2+|<psi_{i,k}|psi_{n,k-e_imu}>|^2)
                    tS[0,imu] += 0.25*(sum(abs(M_a[:,2*imu,i])**2*rE0)+sum(abs(M_a[:,2*imu+1,i])**2*rE0))  # sum_{n\in\complement}(|<psi_{n,k}|psi_{i,k+e_imu}>|^2+|<psi_{n,k}|psi_{i,k-e_imu}>|^2)
                    tS[0,imu] += 0.25*(sum(abs(M_b[:,2*imu,i])**2*rE0)+sum(abs(M_b[:,2*imu+1,i])**2*rE0))  # sum_{n\in\complement}(|<psi_{i,k}|psi_{n,k+e_imu}>|^2+|<psi_{i,k}|psi_{n,k-e_imu}>|^2)
                for i in range(nbs-ibs,nbe-ibs):
                    tQ[i-ds,imu] += 0.25*(sum(abs(M_a[:,2*imu,i])**2)+sum(abs(M_a[:,2*imu+1,i])**2))  # sum_{n\in\complement}(|<psi_{n,k}|psi_{i,k+e_imu}>|^2+|<psi_{n,k}|psi_{i,k-e_imu}>|^2)
                    tQ[i-ds,imu] += 0.25*(sum(abs(M_b[:,2*imu,i])**2)+sum(abs(M_b[:,2*imu+1,i])**2))  # sum_{n\in\complement}(|<psi_{i,k}|psi_{n,k+e_imu}>|^2+|<psi_{i,k}|psi_{n,k-e_imu}>|^2)
                    rE0 = (E_out[:]-E_in[i-ds])/(E_out[:]+E_in[i-ds])
                    rE = rE0 * (1/sqrt(E_in[i-ds]**2+Delta**2)-1/sqrt(E_out[:]**2+Delta**2))
                    tG[i-ds,imu] += 0.25*(sum(abs(M_a[:,2*imu,i])**2*rE)+sum(abs(M_a[:,2*imu+1,i])**2*rE))  # sum_{n\in\complement}(|<psi_{n,k}|psi_{i,k+e_imu}>|^2+|<psi_{n,k}|psi_{i,k-e_imu}>|^2)
                    tG[i-ds,imu] += 0.25*(sum(abs(M_b[:,2*imu,i])**2*rE)+sum(abs(M_b[:,2*imu+1,i])**2*rE))  # sum_{n\in\complement}(|<psi_{i,k}|psi_{n,k+e_imu}>|^2+|<psi_{i,k}|psi_{n,k-e_imu}>|^2)
                    tS[i-ds,imu] += 0.25*(sum(abs(M_a[:,2*imu,i])**2*rE0)+sum(abs(M_a[:,2*imu+1,i])**2*rE0))  # sum_{n\in\complement}(|<psi_{n,k}|psi_{i,k+e_imu}>|^2+|<psi_{n,k}|psi_{i,k-e_imu}>|^2)
                    tS[i-ds,imu] += 0.25*(sum(abs(M_b[:,2*imu,i])**2*rE0)+sum(abs(M_b[:,2*imu+1,i])**2*rE0))  # sum_{n\in\complement}(|<psi_{i,k}|psi_{n,k+e_imu}>|^2+|<psi_{i,k}|psi_{n,k-e_imu}>|^2)
                for i in range(nbe-ibs,ibe-ibs):
                    tQ[-1,imu] += 0.25*(sum(abs(M_a[:,2*imu,i])**2)+sum(abs(M_a[:,2*imu+1,i])**2))  # sum_{n\in\complement}(|<psi_{n,k}|psi_{i,k+e_imu}>|^2+|<psi_{n,k}|psi_{i,k-e_imu}>|^2)
                    tQ[-1,imu] += 0.25*(sum(abs(M_b[:,2*imu,i])**2)+sum(abs(M_b[:,2*imu+1,i])**2))  # sum_{n\in\complement}(|<psi_{i,k}|psi_{n,k+e_imu}>|^2+|<psi_{i,k}|psi_{n,k-e_imu}>|^2)
                    rE0 = (E_out[:]-E_in[-1])/(E_out[:]+E_in[-1])
                    rE = rE0 * (1/sqrt(E_in[-1]**2+Delta**2)-1/sqrt(E_out[:]**2+Delta**2))
                    tG[-1,imu] += 0.25*(sum(abs(M_a[:,2*imu,i])**2*rE)+sum(abs(M_a[:,2*imu+1,i])**2*rE))  # sum_{n\in\complement}(|<psi_{n,k}|psi_{i,k+e_imu}>|^2+|<psi_{n,k}|psi_{i,k-e_imu}>|^2)
                    tG[-1,imu] += 0.25*(sum(abs(M_b[:,2*imu,i])**2*rE)+sum(abs(M_b[:,2*imu+1,i])**2*rE))  # sum_{n\in\complement}(|<psi_{i,k}|psi_{n,k+e_imu}>|^2+|<psi_{i,k}|psi_{n,k-e_imu}>|^2)
                    tS[-1,imu] += 0.25*(sum(abs(M_a[:,2*imu,i])**2*rE0)+sum(abs(M_a[:,2*imu+1,i])**2*rE0))  # sum_{n\in\complement}(|<psi_{n,k}|psi_{i,k+e_imu}>|^2+|<psi_{n,k}|psi_{i,k-e_imu}>|^2)
                    tS[-1,imu] += 0.25*(sum(abs(M_b[:,2*imu,i])**2*rE0)+sum(abs(M_b[:,2*imu+1,i])**2*rE0))  # sum_{n\in\complement}(|<psi_{i,k}|psi_{n,k+e_imu}>|^2+|<psi_{i,k}|psi_{n,k-e_imu}>|^2)

                tC[:,imu] = (dE_imu[:,imu])**2*Delta**2/sqrt(E_in[:]**2+Delta**2)**3
                    
            tR[0 ,:] /= (nbs-ibs+1) # because we added to the bottom band all degenerate bands, we need to normalize.
            tR[-1,:] /= (ibe-nbe+1)  # because we added to the top band all degenerate bands, we need to normalize.
            tQ[0 ,:] /= (nbs-ibs+1) # because we added to the bottom band all degenerate bands, we need to normalize.
            tQ[-1,:] /= (ibe-nbe+1)  # because we added to the top band all degenerate bands, we need to normalize.
            tG[0 ,:] /= (nbs-ibs+1) # because we added to the bottom band all degenerate bands, we need to normalize.
            tG[-1,:] /= (ibe-nbe+1)  # because we added to the top band all degenerate bands, we need to normalize.
            tS[0 ,:] /= (nbs-ibs+1) # because we added to the bottom band all degenerate bands, we need to normalize.
            tS[-1,:] /= (ibe-nbe+1)  # because we added to the top band all degenerate bands, we need to normalize.
            
            for imu in range(3):
                #tR[:,imu] *= (8*pi**2/latgen_Vol)/dst[imu]**2
                tR[:,imu] *= 2.0/dst[imu]**2
                tQ[:,imu] *= 2.0/dst[imu]**2
                tG[:,imu] *= 2.0 * Delta**2*2.0/dst[imu]**2  # I added here 2 because we should some over all m!=n, hence each m and n should appear twice. Check that this is accurate!
                tS[:,imu] *= 2.0/dst[imu]**2
            #print('bands=', [ibs,nbs,nbe,ibe],file=fout)
            for i in range(0,nbe-nbs): #  over all the relevant bands, index i
                print('R[ik='+str(iik)+',i='+str(nbs+i)+']=', tR[i,:].tolist(),file=fout)
                print('Q[ik='+str(iik)+',i='+str(nbs+i)+']=', tQ[i,:].tolist(),file=fout)
                print('S[ik='+str(iik)+',i='+str(nbs+i)+']=', tS[i,:].tolist(),file=fout)
                print('P[ik='+str(iik)+',i='+str(nbs+i)+']=', tG[i,:].tolist(),file=fout)
                print('dE[ik='+str(iik)+',i='+str(nbs+i)+']=', dE_imu[i,:].tolist(),file=fout)
                print('E[ik='+str(iik)+',i='+str(nbs+i)+']:', 'e_i=', E_in.tolist(), 'e_m=', E_out.tolist(), file=fout)
            tQGM.append(tR)
            tQRM.append(tQ)
            tGeD.append(tG)
            tQSM.append(tS)
            tCD.append(tC)
        QGM.append(tQGM)
        QGR.append(tQRM)
        GeD.append(tGeD)
        QSM.append(tQSM)
        CoD.append(tCD)
        
    savetxt('nbs_nbe.dat', array(nbs_nbe,dtype=int))
    fq = open('QGT.dat','w')
    fr = open('QGR.dat','w')
    fp = open('GeD.dat','w')
    fs = open('QSM.dat','w')
    fe = open('eQGT.dat','w')
    fc = open('CoD.dat', 'w')
    dire=['x','y','z']
    for fileh in [fq,fr,fp,fs,fc]:
        print('# %2s %8s' % ('ik', '|d|'), end='',file=fileh)
        for ib in range(len(nbs_nbe)):
            for i in range(nbs_nbe[ib][0],nbs_nbe[ib][1]):
                for imu in range(3):
                    print('%12s '%('band['+str(i)+']:'+dire[imu],), end='', file=fileh)
        print(file=fileh)
        
    kp = klist[0]
    dst=0
    for iik,ik in enumerate(pairs):
        kn = array(klist[ik])
        dst += linalg.norm(k2cartes @ (kn-kp))
        for fileh in [fq,fr,fp,fs,fc]:
            print('%3d %8.5f' % (iik, dst), end='',file=fileh)
        print('%10.5f' % (dst,), end='',file=fe)
        for ib,(nbs,nbe) in enumerate(nbs_nbe):
            for i in range(nbs,nbe):
                print('%12.5f'%(Ebnd[ik,i+nbs0]*H2eV,), end='', file=fe)
                for imu in range(3):
                    print('%12.4f ' % QGM[iik][ib][i-nbs,imu],  end='',file=fq)
                for imu in range(3):
                    print('%12.4f ' % QGR[iik][ib][i-nbs,imu],  end='',file=fr)
                for imu in range(3):
                    print('%12.4f ' % GeD[iik][ib][i-nbs,imu],  end='',file=fp)
                for imu in range(3):
                    print('%12.4f ' % QSM[iik][ib][i-nbs,imu],  end='',file=fs)
                for imu in range(3):
                    print('%12.4f ' % CoD[iik][ib][i-nbs,imu],  end='',file=fc)
        for fileh in [fq,fr,fp,fs,fc,fe]:        
            print(file=fileh)
        kp = kn
    for fileh in [fq,fr,fp,fs,fc]:
        print(file=fileh)

if __name__ == '__main__':
    Delta=5e-3
    if mrank==master:
        #Compute_QGT(Delta,(-1,-1))
        Compute_QGT(Delta)
    # SrVO3: [(0,9),(9,12),(12,20)]
    # MnO:  [(0,3),(3,8)]
    # FeSe: [(0,6),(6,16),(16,23)]
    # LaRuSi: [(0,10),(10,17),(17,25)]
    # Si: [(0,4),(4,10)]
    # LSCO : [(0,16),(16,17)]
    # Bi2212 : [(0,44),(44,50),(51,55)]
    # CaRh2 : [(0,14),(14,18),(18,25)]
