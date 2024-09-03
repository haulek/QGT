#!/usr/bin/env python
# @Copyright 2024 Kristjan Haule
import sys, os, re
from numpy import *
from numpy import linalg
import glob
from fractions import Fraction
from math import gcd

import gwienfile as w2k

def PrintM(A, file=sys.stdout):
    ni,nj = shape(A)
    for i in range(ni):
        for j in range(nj):
            print('%6.3f '% A[i,j], end='',file=file)
        print(file=file)
    
def PrintMC(A, file=sys.stdout):
    ni,nj = shape(A)
    for i in range(ni):
        for j in range(nj):
            print('%10.5f %10.5f '% (A[i,j].real,A[i,j].imag), end='', file=file)
        print(file=file)

def Read_klist(fname):
    fg = open(fname, 'r')
    name_kpoints={}
    kpoints=[]
    dx = 10
    for il,line in enumerate(fg):
        if line[:3]=='END': break
        name = line[:10].split()
        #FORMAT(A10,4I10,3F5.2,A3)  KNAME(kindex), ISX, ISY, ISZ, IDV, WEIGHT(kindex), E1, E2, IPGR(kindex)
        #FORMAT(A10,4I5,3F5.2,A3)   KNAME(kindex), ISX, ISY, ISZ, IDV, WEIGHT(kindex), E1, E2, IPGR(kindex)
        if il==0:
            if len(line[20:30].split()) > 1: dx=5
            #print('dx=', dx)
        kk = [int(line[10+dx*i:10+dx*(i+1)]) for i in range(4)]
        ww = [line[10+dx*4+5*i:10+dx*4+5*(i+1)] for i in range(3)] # weight, e1, e2
        kpoints.append(kk)
        if name:
            legnd=line.split()[0]
            name_kpoints[il]=legnd
    return (array(kpoints),name_kpoints)


def Get_case():
    struct_names = glob.glob('*.struct')
    if len(struct_names)==0:
        print('ERROR : Could not find a *.struct file present')
        sys.exit(0)
    else:
        case = struct_names[0].split('.')[0]
    return case

def Initialize(mode='w', Print=True, out_file='QGT.out'):
    fout = open(out_file, mode)
    case = Get_case()
    if Print: print('case=', case)
    return (case, fout)

    
def Compute_derivative_klist(file_end):
    """Give a list of k-points (path in k-space) generates 7times more k-points, 
       which allow one to calculate derivative in each point. 
       We then execute lapw1 to get eigenvectors.
       The k-points order is [k,k+dkx,k-dkx,k+dky,k-dky,k+dkz,k-dkz]
    """
    def common_denom(i1,i2):
        if ( 2*abs(i1-i2)/(i1+i2) < 0.05 ):  # this is approximation. If i1 and i2 are very close, we do not want to take the product, but approximate
            return max(i1,i2)
        else:
            return int(i1*i2/gcd(i1,i2))
        
    (case, fout) = Initialize('w')
    kpoints, name_kpoints = Read_klist(case+file_end)
    os.rename(case+file_end, case+file_end+'.bak')
    
    nspin=1
    kmr=1  # if we want to increase the G-mesh as compared to case.in1 RKmax
    pwm=2  # plane wave mesh parameter
    
    strc = w2k.Struct(case, fout)
    latgen = w2k.Latgen(strc, fout)
    #latgen.Symoper(strc, fout) I think symope is called in Latgen initialization

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
    
    # creates a list of k-point pairs that we will need to compute QGT
    LD = max(kpoints[:,3])
    
    cart2k = linalg.inv(k2cartes)
    # find the point with minimal distance between two points
    i_min = argmin([linalg.norm(kpoints[i,:3]/kpoints[i,3]-kpoints[i+1,:3]/kpoints[i+1,3]) for i in range(0,len(kpoints)-1)])
    dk = 0.5*linalg.norm(k2cartes @ (kpoints[i_min,:3]/kpoints[i_min,3]-kpoints[i_min+1,:3]/kpoints[i_min+1,3])) # dk in cartesian coordinates
    dkr = array([cart2k @ array([dk,0,0]), cart2k @ array([0,dk,0]), cart2k @ array([0,0,dk])])
    
    print('ik with minimum distance=', i_min, 'and its length in cartesian dk=', 2*dk)
    print('dk in icartesian coordinates=', dkr.tolist())
    print('k2cartes=', k2cartes.tolist())
    print('cart2k=', cart2k.tolist())
    print('k2icartes=', k2icartes.tolist())
    
    fkl = open(case+file_end,'w')
    direction=['x','y','z']
    dn1=0
    npair=[]
    spair=[]
    for ii,kk in enumerate(kpoints):
        ki = kk[:3]/kk[3] # k-list point in semi-cartesian coordinates
        LDk = kk[3]       # common factor for this k-point
        while (LDk<100): LDk *= 2 # we require reasonably large LDk, so that dkr is reasonably well approximated with a fraction
        dkr_LD = dkr*LDk  # step in x,y,z direction multiplied with the common factor, which will be presented as fraction
        # we could just take round of dkr_LD, but we want to do it better.
        # We want to find much more precise common factor
        frk = [[Fraction(str(dkr_LD[i,j])).limit_denominator(50) for j in range(3)] for i in range(3)] # not too large fractions.
        Dk = [common_denom(common_denom(frk[i][0].denominator,frk[i][1].denominator),frk[i][2].denominator)*LDk for i in range(3)]
        Dk = array(Dk,dtype=int)
        pk = array([(ki + dkr[j])*Dk[j] for j in range(3)]) # This is k+dk which should be very close to integer list.
        mk = array([(ki - dkr[j])*Dk[j] for j in range(3)]) # This is k-dk which should be very close to integer list.
        # Now we round it up to an integer. We should check that pk and mk is not too far from integer
        ipk = array(pk.round(),dtype=int) # integer approximation for k+dk
        imk = array(mk.round(),dtype=int) # integer approximation for k-dk
        ddp = array([(pk[j,:]-ipk[j,:])/Dk[j] for j in range(3)]) # checking the difference for k+dk
        ddm = array([(mk[j,:]-imk[j,:])/Dk[j] for j in range(3)]) # checking the difference for k-dk
        addp = linalg.norm(ddp)
        addm = linalg.norm(ddm)
        if (addp > 0.02*dk or addm > 0.02*dk):
            print('WARNING addp=', addp, 'addm=', addm)
            print('pk=', pk, 'while ipk=', ipk, 'addp=', addp)
            print('mk=', mk, 'while imk=', imk, 'addm=', addm)
        #print('ddp=', ddp, linalg.norm(ddp))
        #print('ddm=', ddm, linalg.norm(ddm))
        name=' '
        if ii in name_kpoints: name = name_kpoints[ii]
        print('{:10s}{:10d}{:10d}{:10d}{:10d}'.format(name,kk[0],kk[1],kk[2],kk[3]),file=fkl)
        dn2=1
        for j in range(3):
            dp = linalg.norm(k2cartes @ (ipk[j]/Dk[j]-ki))
            dm = linalg.norm(k2cartes @ (imk[j]/Dk[j]-ki))
            print('{:10s}{:10d}{:10d}{:10d}{:10d}{:20s}{:6f}'.format(' ',ipk[j,0],ipk[j,1],ipk[j,2],Dk[j],'',dp),file=fkl)
            print('{:10s}{:10d}{:10d}{:10d}{:10d}{:20s}{:6f}'.format(' ',imk[j,0],imk[j,1],imk[j,2],Dk[j],'',dm),file=fkl)
            npair.append([dn1,dn1+dn2+0])
            npair.append([dn1,dn1+dn2+1])
            spair.append('k['+str(dn1)+']+d'+direction[j])
            spair.append('k['+str(dn1)+']-d'+direction[j])
            dn2 +=2
            #FORMAT(A10,4I10,3F5.2,A3)
        dn1+=dn2
    print('END', file=fkl)
    fkp = open(case+'.nnkp','w')
    for i in range(len(npair)):
        print('{:5d} {:5d} # {:s}'.format(npair[i][0],npair[i][1],spair[i]),file=fkp)
    fkp.close()
    print('Now execute "x_dmft.py lapw1 --band"')
    
    
if __name__ == '__main__':
    Compute_derivative_klist('.klist_band')
