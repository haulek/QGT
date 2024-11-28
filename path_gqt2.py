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
from datetime import date
from numpy import *
from numpy import linalg
from timeit import default_timer as timer
from scipy import optimize
import itertools
from functools import reduce
from numba import jit
#from numba.typed import List
from scipy import special
import subprocess, shutil
import pickle 
import glob
#from numpy import unravel_index
from fractions import Fraction
from math import gcd

from cmn import Ry2H,H2eV,Ry2eV,H2Ry,Br2Ang
#from inout import InOut
from kqmesh import KQmesh
import mcommon as mcmn
from kohnsham import KohnShamSystem
from planewaves import PlaneWaves
#from matel2band import MatrixElements2Band
#from productbasis import ProductBasis
from kweights import Kweights
import gwienfile as w2k

import radials as rd        # radial wave functions
import lapwcoeff as lapwc   # computes lapw coefficients alm,blm,clm
import gaunt as gn
import for_vxcnn as fvxcn   # matrix elements for Vxcn
import sphbes
from pylab import linalg as la
la_matmul = matmul #la.matmul
from pylab import *
import path_gqt3 as pg3

def PrintM(A, file=sys.stdout):
    ni,nj = shape(A)
    for i in range(ni):
        for j in range(nj):
            print('%6.3f '% A[i,j], end='',file=file)
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

def Find_EF(case):
    scf = open(case+'.scf','r').readlines()
    for line in scf[::-1]:
        if line[:4]==':FER':
            EF = float(line.split()[-1])
            return EF
    return 0

class Orbitals:
    """ Class which can read dmft.indmfl file and set up rotation matrices and local axes for all atoms.
    For orbitals which are not in indmfl file, it sets up cubic harmonics in local axis, which is given
    in wien2k structure file.
    """
    def __init__(self, strc, in1): #, in1_nlomax):
        s2 = sqrt(2.)
        T2Cs = [[1]]
        #           pz           px                 py
        T2Cp = [ [0,1,0], [1/s2, 0, -1/s2], [-1j/s2, 0, -1j/s2]]
        #           z2          x2-y2             xz                 yz                xy
        T2Cd = [[0,0,1,0,0],[1/s2,0,0,0,1/s2],[0,1/s2,0,-1/s2,0],[0,1j/s2,0,1j/s2,0],[1j/s2,0,0,0,-1j/s2]]
        #       fz3              fxz2                  fyz2                    fz(x2-y2)             fxyz                     fx(x2-3y2)             fy(3x2-y2)
        T2Cf = [[0,0,0,1,0,0,0],[0,0,1/s2,0,-1/s2,0,0],[0,0,1j/s2,0,1j/s2,0,0],[0,1/s2,0,0,0,1/s2,0],[0,1j/s2,0,0,0,-1j/s2,0],[1/s2,0,0,0,0,0,-1/s2],[1j/s2,0,0,0,0,0,1j/s2]]
        self.S2C_T = [ array(T2Cs).T, array(T2Cp).T, array(T2Cd).T, array(T2Cf).T]
        self.S2C_N = [['s'],['pz','px','py'],['z2','x2-y2','xz','yz','xy'],['fz3','fxz2','fyz2','fz(x2-y2)','fxyz','fx(x2-3y2)','fy(3x2-y2)']]

        self.aname = []
        for iat in range(len(strc.mult)):
            nam = strc.aname[iat].split()[0] # If space appears in struct file, such name is not working in wannier90, so remove anything after space.
            if strc.mult[iat]==1:
                self.aname.append( nam )
            else:
                self.aname += [ nam+str(ieq+1) for ieq in range(strc.mult[iat])]

        if in1.nlomax > 0:
            # If we have local orbitals, we will increase np_lo to 2
            self.np_lo = 2
        else:
            self.np_lo = 1

    def ReadIndmflFile(self, case):
        def divmodulo(x,n):
            "We want to take modulo and divide in fortran way, so that it is compatible with fortran code"
            return ( sign(x)* int(abs(x)/n) , sign(x)*mod(abs(x),n))
        fi = open(case+'.indmfl','r')
        lines = fi.readlines()
        fi.close()
        lines = [line.split('#')[0].strip() for line in lines] # strip comments
        lines = [line for line in lines if line]  # strip blank lines & create generator expression
        itr=0
        dat = lines[itr].split(); itr+=1
        #print('dat=', dat)
        hybr_emin, hybr_emax, Qrenorm, projector = float(dat[0]), float(dat[1]), int(dat[2]), int(dat[3])
        dat = lines[itr].split(); itr+=1
        #print('dat=', dat)
        matsubara, broadc, broadnc, om_npts, om_emin, om_emax = int(dat[0]), float(dat[1]), float(dat[2]), int(dat[3]), float(dat[4]), float(dat[5])
        if projector>=5:
            hybr_emin = int(hybr_emin)
            hybr_emax = int(hybr_emax)

        self.atoms={}
        self.icix={}
        self.rotloc={}
        
        self.Lsa=[]
        self.icixa=[]
        natom = int(lines[itr]); itr+=1
        for i in range(natom):
            dat=lines[itr].split()
            itr+=1
            iatom, nL, locrot_shift = [int(x) for x in dat[:3]]
            Rmt2=0
            if len(dat)>3:
                Rmt2 = float(dat[3])
            (shift,locrot) = divmodulo(locrot_shift,3)
            if locrot<0:
                if locrot==-2:
                    locrot=3*nL
                else:
                    locrot=3
            Ls, qsplits, icx = zeros(nL,dtype=int), zeros(nL,dtype=int), zeros(nL,dtype=int)
            for il in range(nL):
                (Ls[il], qsplits[il], icx[il]) = map(int, lines[itr].split()[:3])
                itr+=1
            self.Lsa.append( Ls )
            self.icixa.append( icx )
            new_xyz=[]
            for loro in range(abs(locrot)):
                new_xyz.append( [float(x) for x in lines[itr].split()] )
                itr += 1
            if new_xyz:
                self.rotloc[iatom-1] = new_xyz                
            shift_vec = None
            if shift:
                shift_vec = [float(x) for x in lines[itr].split()]
                itr += 1
            #print( 'new_xyz=', new_xyz)
            #print( 'shift_vec=', shift_vec)
            
            #self.locrot[iatom-1] = (locrot, shift)
            self.atoms[iatom-1] = (nL, locrot_shift, shift_vec, Rmt2)
            for ix, L, qsplit in zip(icx, Ls, qsplits):
                if ix in self.icix:
                    self.icix[ix] += [(iatom-1, L, qsplit)]
                else:
                    self.icix[ix] = [(iatom-1, L, qsplit)]
            
        self.legends={}
        self.legend={}
        self.siginds={}
        self.cftrans={}
        # read the big block of siginds and cftrans
        ncp, maxdim, maxsize = [int(e) for e in lines[itr].split()]
        itr+=1
        for i in range(ncp):
            icp, dim, size = [int(e) for e in lines[itr].split()]
            itr+=1
            self.legends[icp] = lines[itr].split("'")[1::2]
            #print('legends=', self.legends[icp])
            (iatom,l,qsplit) = self.icix[icp][0]
            self.legend[(iatom,l)] = self.legends[icp]
            #print('legend[('+str(iatom)+','+str(l)+')]=',self.legend[(iatom,l)])
                                  
            itr+=1
            sigi=[]
            for row in range(dim):
                sigi.append([int(e) for e in lines[itr].split()])
                itr+=1
            self.siginds[icp] = array(sigi,dtype=int)
            raw_cftrans=[]
            for row in range(dim):
                raw_cftrans.append([float(e) for e in lines[itr].split()])
                itr+=1
            raw_cftrans=array(raw_cftrans)
            self.cftrans[icp] = raw_cftrans[:,0::2] + raw_cftrans[:,1::2]*1j
        #print('atoms=', self.atoms)
        #print('icix=', self.icix)
        #print('rotloc=', self.rotloc)
        #print('Lsa=', self.Lsa)
        #print('icixa=', self.icixa)
        #print('legends=', self.legends)
        #print('sigind=', self.siginds)
        #print('cftrans=', self.cftrans)

    def FromIndmfl(self):
        ndf = len(self.aname)
        #ndf = sum(strc.mult)
        np_lo = self.np_lo
        ip_lo=0
        orbts=[]
        CFx={}
        Sigind={}
        for icx in self.icix:
            cixs = self.icix[icx]
            sigind = self.siginds[icx]
            cFx = self.cftrans[icx]
            #print('Sigind=', sigind)
            ii=0
            for x in cixs:
                idf,l = x[0:2]
                orb = [idf + ndf*ip_lo + ndf*np_lo*(l**2 + mr) for mr in range(2*l+1) if sigind[ii+mr,ii+mr]]
                orbts += orb
                #print('idf=', idf, 'l=', l, 'ip_lo=', ip_lo, 'ndf=', ndf, 'ind=', orb)
                CFx[(idf,l)]=cFx
                #print('l=', l, 'ii=', ii, 'sigind=', sigind)
                Sigind[(idf,l)]= [sigind[ii+mr,ii+mr] for mr in range(2*l+1)]
                ii+=2*l+1
        return (orbts, CFx)
            
    def Convert_iorb2text(self, orbs):
        lmaxp1 = 4
        ndf = len(self.aname)
        #print('nn=', lmaxp1**2*self.np_lo*ndf, 'lmaxp1=', lmaxp1, 'np_lo=', self.np_lo, 'ndf=', ndf)
        names = zeros( (lmaxp1**2,self.np_lo,ndf), dtype=(unicode_, 20))
        for l in range(lmaxp1):
            for m in range(-l,l+1):
                lm = l**2 + l + m
                for idf in range(ndf):
                    mr_name = str(l+m+1)
                    #print('(idf,l)=', (idf,l), 'keys=', self.legend.keys())
                    if (idf,l) in self.legend:
                        mr_name = self.legend[(idf,l)][l+m]
                        #print('mr_name=',mr_name)
                    nm1 = self.aname[idf]+':l='+str(l)+',mr='+mr_name
                    names[lm,0,idf] = nm1
                    if self.np_lo>1 :
                        nm2 = self.aname[idf]+':l='+str(l)+',mr='+mr_name #+',r=2'
                        names[lm,1,idf] = nm2
        names = reshape(names, lmaxp1**2*self.np_lo*ndf)

        #for i in range(len(names)):
        #    print('names['+str(i)+']=', names[i])
        #print('orbs=', orbs)
        #print('nl_lo=', self.np_lo, 'N=', lmaxp1**2*self.np_lo*ndf)
        orb_names = [names[i] for i in orbs]
        return orb_names
    
    def FindRelevantOrbitals(self,nbs,nbe,k_ind,rotloc,strc,latgen,ks,kqm,pw,in1,radf,fout,debug=False):
        lmaxp1 = min(4,in1.nt)
        lomaxp1 = shape(in1.nlo)[0]
        ndf = sum(strc.mult)
        if in1.nlomax > 0:
            # If we have local orbitals, we will increase np_lo to 2
            self.np_lo = 2
        else:
            self.np_lo = 1
        
        alfr = zeros((nbe-nbs,lmaxp1**2,self.np_lo,ndf), dtype=complex)
        asmr = zeros((lmaxp1**2*self.np_lo*ndf), dtype=float)
        self.largest_ilo = -ones((ndf,lomaxp1),dtype=int)
        nextra=0
        orbs=[]
        for iik,ik in enumerate(k_ind):
            kil = ks.klist[ik]
            Aeigk = array(ks.all_As[ik][nbs:nbe,:], dtype=complex)   # eigenvector from vector file
            
            #alm,blm,clm = lapwc.gap2_set_lapwcoef(kil, ks.indgkir[ik], 1, True, ks.nv[ik], pw.gindex, radf.abcelo[0], strc.rmt, strc.vpos, strc.mult, radf.umt[0], rotloc, latgen.trotij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)

            timat_ik, tau_ik = identity(3), array([0,0,0])
            alm,blm,clm = lapwc.dmft1_set_lapwcoef(False,1,True,kil,kil,timat_ik,tau_ik,ks.indgkir[ik],ks.nv[ik],pw.gindex,radf.abcelo[0],strc.rmt,strc.vpos,strc.mult,radf.umt[0],
                                                   rotloc,latgen.rotij,latgen.tauij,latgen.Vol,kqm.k2cartes,in1.nLO_at,in1.nlo,in1.lapw,in1.nlomax)
            
            (ngi,nLOmax,ntnt,ndf) = shape(clm)
            (ngk,ntnt,ndf) = shape(alm)
            (nbmax,ngi2) = shape(Aeigk)
            # And now change alm,blm,clm to band basis, which we call alfa,beta,gama
            alfa = reshape( la_matmul(Aeigk, reshape(alm, (ngk,ntnt*ndf)) ), (nbmax,ntnt,ndf) )
            if self.np_lo > 1:
                beta = reshape( la_matmul(Aeigk, reshape(blm, (ngi,ntnt*ndf)) ), (nbmax,ntnt,ndf) )
                gama = reshape( la_matmul(Aeigk, reshape(clm, (ngi,ntnt*ndf*nLOmax)) ), (nbmax,nLOmax,ntnt,ndf) )
                idf2iat = concatenate([[iat]*strc.mult[iat] for iat in range(strc.nat)])
                for idf in range(ndf):
                    iat = idf2iat[idf]
                    for l in range(lomaxp1):
                        r = zeros(in1.nLO_at[l,iat])
                        for ilo in range(in1.lapw[l,iat],in1.nLO_at[l,iat]):
                            s2 = radf.umtlo[0,iat,l,ilo,2]  # <u_l|u_lo>
                            s3 = radf.umtlo[0,iat,l,ilo,3]  # <udot_l|u_lo>
                            alfa[:,l**2:(l+1)**2,idf] += s2 * gama[:,ilo,l**2:(l+1)**2,idf]  # 
                            cg = sqrt(1-s2**2)
                            cb = s3/cg
                            gama[:,ilo,l**2:(l+1)**2,idf] = cg*gama[:,ilo,l**2:(l+1)**2,idf] + cb*beta[:,l**2:(l+1)**2,idf]
                            if iik==0: r[ilo] = sum(abs(gama[:,ilo,l**2:(l+1)**2,idf])**2)
                        if iik==0 and in1.nLO_at[l,iat]:
                            print('idf=', idf, 'l=', l, 'ilo=', ilo, 'r=', r, file=fout)
                            self.largest_ilo[idf,l] = argmax(r)
                            print('largest_ilo[idf='+str(idf)+',l='+str(l)+']='+str(self.largest_ilo[idf,l]), file=fout)
                            
            for idf in range(ndf):
                for l in range(lmaxp1):
                    alfr[:,l**2:(l+1)**2,0,idf] = alfa[:,l**2:(l+1)**2,idf] @ self.S2C_T[l].T
                    if self.largest_ilo[idf,l]>=0 :
                        ilo = self.largest_ilo[idf,l]
                        alfr[:,l**2:(l+1)**2,1,idf] = gama[:,ilo,l**2:(l+1)**2,idf] @ self.S2C_T[l].T

            alfr = reshape(alfr,(nbmax,lmaxp1**2*self.np_lo*ndf))
            asm = sum(abs(alfr)**2, axis=0)
            asmr += asm
            alfr = reshape(alfr,(nbmax,lmaxp1**2,self.np_lo,ndf))
            
        ind = sorted( range(len(asm)), key=lambda x: -asmr[x] )
        #print('nbmax=', nbmax, 'ind=', ind)
        orbs = ind[:nbmax]
        #print('orbs=', orbs)
        #print('nn=', lmaxp1**2*self.np_lo*ndf, 'lmaxp1=', lmaxp1, 'np_lo=', self.np_lo, 'ndf=', ndf)
        #print('orbitals to be used in projection to wanniers=', orbs, file=fout)
        orb_names = self.Convert_iorb2text(orbs)
        #for i in range(len(orb_names)):
        #    print(i,orb_names[i], file=fout)
        #print('w=', asmr[orbs])
        #for ii,i in enumerate(orbs):
        #    print(ii,i,asmr[i])
            
        return (orbs, orb_names, asmr[orbs])

def Compute_and_Save_Projection_amn(nbs,nbe,rotloc,k_ind, case, orbits, orbs, CFx, strc, latgen, ks, kqm, pw, in1, radf, fout, debug=False):
    fo_amn = open(case+'.amn', 'w')
    print('# code path_gqt2py', date.today().strftime("%B %d, %Y"), file=fo_amn)
    print(nbe-nbs, len(k_ind), len(orbs), orbits.Convert_iorb2text(orbs), file=fo_amn)

    tfo_amn = open(case+'.amn2', 'w')
    print('# code path_gqt2py', date.today().strftime("%B %d, %Y"), file=tfo_amn)
    
    lmaxp1 = min(4,in1.nt)
    lomaxp1 = shape(in1.nlo)[0]
    ndf = sum(strc.mult)

    icartes2f = linalg.inv(kqm.k2icartes)
    print('icartes2f=', icartes2f, file=fout)
    
    #if in1.nlomax > 0:
    #    # If we have local orbitals, we will increase np_lo to 2
    #    self.np_lo = 2
    #else:
    #    self.np_lo = 1
    self_np_lo = orbits.np_lo
    largest_ilo = -ones((ndf,lomaxp1),dtype=int)
    alfr = zeros((nbe-nbs,lmaxp1**2,self_np_lo,ndf), dtype=complex)
    asmr = zeros((lmaxp1**2*self_np_lo*ndf), dtype=float)
    print('The singular values of projection to bands <chi|psi>: ', file=fout)
    DMFT1 = True
    for iik,ik in enumerate(k_ind):#range(len(kqm.klist)):
        kil = ks.klist[ik]
        Aeigk = array(ks.all_As[ik][nbs:nbe,:], dtype=complex)   # eigenvector from vector file
        if DMFT1:
            timat_ik, tau_ik = identity(3), array([0,0,0])
            alm,blm,clm = lapwc.dmft1_set_lapwcoef(False,1,True,kil,kil,timat_ik,tau_ik,ks.indgkir[ik],ks.nv[ik],pw.gindex,radf.abcelo[0],strc.rmt,strc.vpos,strc.mult,radf.umt[0],
                                                    rotloc,latgen.rotij,latgen.tauij,latgen.Vol,kqm.k2cartes,in1.nLO_at,in1.nlo,in1.lapw,in1.nlomax)
        else:
            alm,blm,clm = lapwc.gap2_set_lapwcoef(kil, ks.indgkir[ik], 1, True, ks.nv[ik], pw.gindex, radf.abcelo[0], strc.rmt, strc.vpos, strc.mult, radf.umt[0], rotloc, latgen.trotij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
        
        (ngi,nLOmax,ntnt,ndf) = shape(clm)
        (ngk,ntnt,ndf) = shape(alm)
        (nbmax,ngi2) = shape(Aeigk)
        # And now change alm,blm,clm to band basis, which we call alfa,beta,gama
        alfa = reshape( la_matmul(Aeigk, reshape(alm, (ngk,ntnt*ndf)) ), (nbmax,ntnt,ndf) )
        # We will use two functions, the head and the orthogonalized local orbital, i.e.,
        #    |1> == |u_l> and |2>= (|u_lo>-|u_l><u_l|u_lo>)/sqrt(1-<u_l|u_lo>^2)
        # Note that <1|2>=0 and <2|2>=1
        if self_np_lo > 1:
            beta = reshape( la_matmul(Aeigk, reshape(blm, (ngk,ntnt*ndf)) ), (nbmax,ntnt,ndf) )
            gama = reshape( la_matmul(Aeigk, reshape(clm, (ngk,ntnt*ndf*nLOmax)) ), (nbmax,nLOmax,ntnt,ndf) )
            idf2iat = concatenate([[iat]*strc.mult[iat] for iat in range(strc.nat)])
            for idf in range(ndf):
                iat = idf2iat[idf]
                for l in range(lomaxp1):
                    r = zeros(in1.nLO_at[l,iat])
                    for ilo in range(in1.lapw[l,iat],in1.nLO_at[l,iat]):
                        s2 = radf.umtlo[0,iat,l,ilo,2]  # <u_l|u_lo>
                        s3 = radf.umtlo[0,iat,l,ilo,3]  # <udot_l|u_lo>
                        alfa[:,l**2:(l+1)**2,idf] += s2 * gama[:,ilo,l**2:(l+1)**2,idf]  # 
                        cg = sqrt(1-s2**2)
                        cb = s3/cg
                        gama[:,ilo,l**2:(l+1)**2,idf] = cg*gama[:,ilo,l**2:(l+1)**2,idf] + cb*beta[:,l**2:(l+1)**2,idf]
                        if iik==0: r[ilo] = sum(abs(gama[:,ilo,l**2:(l+1)**2,idf])**2)
                    if iik==0 and in1.nLO_at[l,iat]:
                        largest_ilo[idf,l] = argmax(r)
                        
        for idf in range(ndf):
            for l in range(lmaxp1):
                S2C_T = orbits.S2C_T[l].T
                if (idf,l) in CFx:
                    S2C_T = CFx[(idf,l)].T
                alfr[:,l**2:(l+1)**2,0,idf] = alfa[:,l**2:(l+1)**2,idf] @ S2C_T
                if largest_ilo[idf,l]>=0 :
                    ilo = largest_ilo[idf,l]
                    alfr[:,l**2:(l+1)**2,1,idf] = gama[:,ilo,l**2:(l+1)**2,idf] @ S2C_T
        
        alfr = reshape(alfr,(nbmax,lmaxp1**2*self_np_lo*ndf))
        
        if iik==0:
            asm = sum(abs(alfr)**2, axis=0) # sum over all bands, and just retaining orbital indices asm[iorb]
            ind = sorted( range(len(asm)), key=lambda x: -asm[x] )
            torbs = ind[:nbe-nbs]
            print(nbe-nbs, len(k_ind), len(torbs),  orbits.Convert_iorb2text(torbs), file=tfo_amn)

        psi_chi = conj(alfr[:,orbs])    # psi_chi[nbnd,norbs] using fancy indexing, just for orbs in the indmfl list
        tpsi_chi = conj(alfr[:,torbs])    # psi_chi[nbnd,norbs] using fancy indexing, just for orbs in the indmfl list
        
        #asm = sum(abs(alfr)**2, axis=0) # sum over all bands, and just retaining orbital indices asm[iorb]
        #asmr += asm
        alfr = reshape(alfr,(nbmax,lmaxp1**2,self_np_lo,ndf))
        
        for n in range(len(orbs)):
            for m in range(nbmax):
                print('%4d%4d %5d %25.12f%25.12f' % (m+1, n+1, iik+1, psi_chi[m,n].real, psi_chi[m,n].imag), file=fo_amn)
        for n in range(len(torbs)):
            for m in range(nbmax):
                print('%4d%4d %5d %25.12f%25.12f' % (m+1, n+1, iik+1, tpsi_chi[m,n].real, tpsi_chi[m,n].imag), file=tfo_amn)

        u, s, vh = linalg.svd(psi_chi, full_matrices=True)  # uv[nbnd,norb]
        print('SVD: ik=%3d'%ik, 'k=(%5.3f,%5.3f,%5.3f)' % tuple(icartes2f@kil), 's=', list(s), file=fout)

    fo_amn.close()
    tfo_amn.close()
    

def Compute_and_Save_BandOverlap_Mmn(pairs, pair_umklap, s_distances, s_idistance,nbs,nbe,s_Irredk, case, strc, latgen, ks, kqm, pw, in1, radf, fout, DMFT1=False, debug=False):
    def FindInArray(ang, angles):
        for ii,a in enumerate(angles):
            if linalg.norm(ang-a) < 1e-6:
                return ii
        return -1
    
    nspin,isp=1,0  # need to do further work for magnetic type calculation
    maxlp1 = min(in1.nt,7)
    # We are creating a product-like basis, which is serving the plane wave expansion e^{-i*b*r} = 4*pi*i^l * j_l(|b|*r) Y_{lm}^*(-b) Y_{lm}(r), here l is big_l, and |b| is distance
    big_l = array([repeat( range(maxlp1),  len(s_distances)) for iat in range(strc.nat)])    # index for l [0,0,0,1,1,1,....] each l is repeated as many times as there are distances.
    big_d = array([concatenate([ list(range(len(s_distances))) for i in range(maxlp1)]) for iat in range(strc.nat)]) # index for distances
    
    print('big_l for using GW projection <u_big_l|psi^*_{k-q}psi_k>', file=fout)
    for iat in range(strc.nat):
        for irm in range(len(big_l[iat])):
            print(irm, 'iat=', iat, 'l=', big_l[iat,irm], 'id=', big_d[iat,irm], file=fout)
    
    j_l_br = [[ [] for irm in range(len(big_l[iat]))] for iat in range(strc.nat)]
    for iat in range(strc.nat):
        rp, dh, npt = strc.radial_mesh(iat)  # logarithmic radial mesh
        for irm,L in enumerate(big_l[iat,:]):
            ab = s_distances[big_d[iat,irm]]        # this is |b|
            ur = special.spherical_jn(L,ab*rp)    # j_l(|b|r)
            j_l_br[iat][irm] = ur              # saving j_l(|b|r)
    
    lomaxp1 = shape(in1.nlo)[0]
    num_radial=[]
    for iat in range(strc.nat):
        tnum_radial = 2*in1.nt
        for l in range(lomaxp1):
            tnum_radial += len(in1.nLO_at_ind[iat][l])
        num_radial.append(tnum_radial)
    # Computing all necessary radial integrals <j_lb(br)|u_l2 u_l1>
    s3r = zeros( (len(big_l[0]),amax(num_radial),amax(num_radial),strc.nat), order='F' )
    for iat in range(strc.nat):
        rp, dh, npt = strc.radial_mesh(iat)  # logarithmic radial mesh
        rwf_all = []
        lrw_all = []
        # ul functions
        for l in range(in1.nt):
            rwf_all.append( (radf.ul[isp,iat,l,:npt],radf.us[isp,iat,l,:npt]) )
            lrw_all.append((l,1))
        # energy derivative ul_dot functions
        for l in range(in1.nt):
            rwf_all.append( (radf.udot[isp,iat,l,:npt],radf.usdot[isp,iat,l,:npt]) )
            lrw_all.append((l,2))
        # LO local orbitals
        for l in range(lomaxp1):
            for ilo in in1.nLO_at_ind[iat][l]:
                rwf_all.append( (radf.ulo[isp,iat,l,ilo,:npt],radf.uslo[isp,iat,l,ilo,:npt]) )
                lrw_all.append((l,3))
        # now performing the integration
        for ir2 in range(len(lrw_all)):
            l2, it2 = lrw_all[ir2]
            a2, b2 = rwf_all[ir2]
            for ir1 in range(ir2+1):
                l1, it1 = lrw_all[ir1]
                a1, b1 = rwf_all[ir1]
                a3 = a1[:] * a2[:] #/ rp[:]
                b3 = b1[:] * b2[:] #/ rp[:]
                for irm in range(len(big_l[iat])):  # over all product functions
                    L = big_l[iat,irm]         # L of the product function
                    if L < abs(l1-l2) or L > (l1+l2): continue # triangular inequality violated
                    ## s3r == < u-product-basis_{irm}| u_{a1,l1} u_{a2,l2} > 
                    rint = rd.rint13g(strc.rel, j_l_br[iat][irm], j_l_br[iat][irm], a3, b3, dh, npt, strc.r0[iat])
                    #print('ir2=', ir2, 'ir1=', ir1, 'irm=', irm, 'int=', rint)
                    s3r[irm,ir1,ir2,iat] = rint
                    s3r[irm,ir2,ir1,iat] = rint
        if True:
            orb_info = [' core',' u_l ','u_dot',' lo  ']
            print((' '*5)+'Integrals <j_L(br)| u_(l1)*u_(l2)> for atom  %10s' % (strc.aname[iat],), file=fout)
            print((' '*13)+'N   L   l1  u_   l2 u_        <v | u*u>', file=fout)
            for irm in range(len(big_l[iat])):  # over all product functions
                L = big_l[iat,irm]             # L of the product function
                for ir2 in range(len(lrw_all)):
                    l2, it2 = lrw_all[ir2]
                    for ir1 in range(ir2+1):
                        l1, it1 = lrw_all[ir1]
                        if abs(s3r[irm,ir1,ir2,iat])>1e-10:
                            print(('%4d%4d'+(' '*2)+('%4d'*3)+' %s%4d %s%19.11e') % (ir1,ir2,irm,L,l1,orb_info[it1],l2,orb_info[it2],s3r[irm,ir1,ir2,iat]), file=fout)

    nmix = array([ maxlp1*len(s_distances) for iat in range(strc.nat)], dtype=int)
    loctmatsize  = sum([sum([(2*L+1)*strc.mult[iat] for L in big_l[iat]]) for iat in range(strc.nat)])
    ncore = zeros(strc.nat, dtype=int)  # we do not care about core here, so just set to zero
    cgcoef = gn.cmp_all_gaunt(in1.nt)   # gaunt coefficients
    
    cfac = []  # precompute 4*pi*(-i)^l, but we repeat the entry (2*lb+1) times, because of degeneracy with respect to m
    for lb in range(maxlp1):
        cfac += [4*pi*(-1j)**lb] * (2*lb+1)
    cfac = array(cfac)

    (nnkp, nntot) = len(pairs),len(pairs[0])
    G_umklap = zeros(3,dtype=int)
    
    # It turns out there are only a few possible angles Y_lm(b), for which Y_lm needs to be evaluates
    # We first find all possible angles of vector b.
    b_angles = []
    b_pair = zeros((nnkp,nntot), dtype=int)
    for iik,ik in enumerate(pairs):
        for i in range(len(pairs[ik])):
            jk = pairs[ik][i]
            k1 = array(ks.klist[ik])
            k2 = array(ks.klist[jk])
            if s_Irredk:
                G_umklap = dot(kqm.k2icartes, pair_umklap[iik,i,:])     # umklap G
            b = k2-k1+G_umklap # |k2-k1| is distance between the two k-points
            ab = linalg.norm(b)
            b_ang = zeros(3)
            if ab!=0:
                b_ang = b/ab
            ic = FindInArray( b_ang, b_angles)
            if ic<0:
                b_pair[iik,i] = len(b_angles)
                b_angles.append(b_ang)
            else:
                b_pair[iik,i] = ic
    
    print('Found the following angles for b=k2-k1 vector:', file=fout)
    for i in range(len(b_angles)):
        print(i, b_angles[i], file=fout)
    
    idf = -1
    imm = -1
    ndf = sum(strc.mult)
    Ylmb = zeros((ndf,len(b_angles), maxlp1**2), dtype=complex)
    for iat in range(strc.nat):
        for ieq in range(strc.mult[iat]):
            idf += 1
            Ttot = strc.rotloc[iat,:,:] @ latgen.trotij[idf,:,:].T @ kqm.k2cartes[:,:] # All transformations that we need to evaluate Y_lm(b)
            for i,b in enumerate(b_angles):
                Ylmb[idf,i,:] = sphbes.ylm( dot(Ttot, -b), maxlp1-1)*cfac  # 4*pi(-i)^l * Y_lm(-b), which is e^{-i*b*r}/Y^*_{lm}(r)
                
    mpwipw = zeros((1,len(pw.ipwint)),dtype=complex) # will contain Integrate[e^{iG*r},{r in interstitials}]
    mpwipw[0,:] = pw.ipwint # mpwipw[0,ipw] = <0|G_{ipw}>_{int}=Integrate[e^{i*(G_{ipw}r)},{r in interstitial}]

    #print('writing: ', case+'.mmn', file=fout)
    #fo_mmn = open(case+'.mmn', 'w')
    #print('# code PyGW:wannier.py', date.today().strftime("%B %d, %Y"), file=fo_mmn)
    #print(s_nbe-s_nbs, len(kqm.klist), nntot, file=fo_mmn)
    
    V_rmt = 0
    for iat in range(strc.nat):
        for ieq in range(strc.mult[iat]):
            V_rmt += 4*pi*strc.rmt[iat]**3/3
    print('V_rmt/V=', V_rmt/latgen.Vol, 'V_I/V=', 1-V_rmt/latgen.Vol, 'V_rmt=', V_rmt, 'V=', latgen.Vol, file=fout)

    ikstart,ikend,sendcounts,displacements = mcmn.mpiSplitArray(mrank, msize, nnkp )
    key_pairs = list(pairs.keys())
    #print('key_pairs=', key_pairs)
    print('processor rank=', mrank, 'with size=', msize, 'will do ikstart=', ikstart, 'ikend=', ikend, 'displacements=',displacements, file=fout)
    fout.flush()

    M_all=zeros((ikend-ikstart,nntot,nbe-nbs,nbe-nbs),dtype=complex)
    #R=zeros((ikend-ikstart,3,nbe-nbs))
    idf = -1
    for iik in range(ikstart,ikend):
        #ibs,ibe = nbs_nbe[iik]
        ik = key_pairs[iik]
        kil = array(ks.klist[ik]) # kqm.klist[ik,:]/kqm.LCM  # k1==kil in semi-cartesian form
        if s_Irredk:
            # First create alpha, beta, gamma for k
            irk = kqm.kii_ind[ik]          # the index to the correspodning irreducible k-point
            Aeigk = array(ks.all_As[irk][nbs:nbe,:], dtype=complex)   # eigenvector from vector file
            if not DMFT1:
                if kqm.k_ind[irk] != ik:                       # not irreducible
                    Aeigk *= exp( 2*pi*1j * ks.phase_arg[ik][:])  # adding phase: e^{i*tau[isym]*(k_irr+G_irr)}, where tau[isym] is the part of the group operation from irreducible to reducible k-point
                alm,blm,clm = lapwc.gap2_set_lapwcoef(kil, ks.indgk[ik], 1, True, ks.nv[irk], pw.gindex, radf.abcelo[isp], strc.rmt, strc.vpos, strc.mult, radf.umt[isp], strc.rotloc, latgen.trotij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
            else:
                isym = kqm.iksym[ik]
                timat_ik, tau_ik = strc.timat[isym,:,:].T, strc.tau[isym,:]
                kirr = array(kqm.kirlist[irk,:])/float(kqm.LCM)
                alm,blm,clm = lapwc.dmft1_set_lapwcoef(False, 1, True, kil, kirr, timat_ik, tau_ik, ks.indgkir[irk], ks.nv[irk], pw.gindex, radf.abcelo[0], strc.rmt, strc.vpos, strc.mult, radf.umt[0], strc.rotloc, latgen.rotij, latgen.tauij, latgen.Vol,  kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
        else:
            Aeigk = array(ks.all_As[ik][nbs:nbe,:], dtype=complex)   # eigenvector from vector file
            alm,blm,clm = lapwc.gap2_set_lapwcoef(kil, ks.indgkir[ik], 1, True, ks.nv[ik], pw.gindex, radf.abcelo[isp], strc.rmt, strc.vpos, strc.mult, radf.umt[isp], strc.rotloc, latgen.trotij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
            
        (ngi,nLOmax,ntnt,ndf) = shape(clm)
        (ngi,ntnt,ndf) = shape(alm)
        (nbmax,ngi2) = shape(Aeigk)
        # And now change alm,blm,clm to band basis, which we call alfa,beta,gama
        alfa = reshape( la_matmul(Aeigk, reshape(alm, (ngi,ntnt*ndf)) ), (nbmax,ntnt,ndf) )
        beta = reshape( la_matmul(Aeigk, reshape(blm, (ngi,ntnt*ndf)) ), (nbmax,ntnt,ndf) )
        if in1.nlomax > 0:
            gama = reshape( la_matmul(Aeigk, reshape(clm, (ngi,ntnt*ndf*nLOmax)) ), (nbmax,nLOmax,ntnt,ndf) )
        else:
            gama = zeros((1,1,1,1),dtype=complex,order='F') # can not have zero size. 

        
        if DMFT1 and s_Irredk and kqm.k_ind[irk] != ik:                       # not irreducible
            # For interstitials we need to transform now, because we did not transform the eigenvectors before.
            Aeigk *= exp( 2*pi*1j * ks.phase_arg[ik][:])  # phase for reducible: e^{i*tau[isym]*(k_irr+G_irr)}, where tau[isym] is the part of the group operation from irreducible to reducible k-point
        
        if debug:
            print('alfa,beta,gama=', file=fout)
            for ie in range(shape(alfa)[0]):
                for lm in range(shape(alfa)[1]):
                    print('ie=%3d lm=%3d alfa=%14.10f%14.10f beta=%14.10f%14.10f' % (ie+1,lm+1,alfa[ie,lm,0].real, alfa[ie,lm,0].imag, beta[ie,lm,0].real, beta[ie,lm,0].imag), file=fout)
        #M_c = zeros((nntot,ibe-ibs,ibe-ibs),dtype=complex)
        for idk,jk in enumerate(pairs[ik]):
            #jk = pair[ik,idk]
            # M_{m,n}   = < psi_{m,k} | e^{-i b r} |psi_{n,k+b}>
            # M_{m,n}^* = < psi_{n,k+b}| e^{i b r} |psi_{m,k}>
            # M_{m,n}^* = < psi_{n,k-q}| e^{-i q r}|psi_{m,k}> where q=-b. The last form is equivalent to our definition of M matrix element in GW.
            # k2 = k1 + b - G_umklap => b = k2-k1+G_umklapp
            #kjl = kqm.klist[jk,:]/kqm.LCM  # k2==kjl=k+b in semi-cartesian form
            kjl = array(ks.klist[jk])
            if s_Irredk:
                G_umklap = array(dot(kqm.k2icartes,pair_umklap[iik,idk,:]), dtype=int)
            bl = kjl-kil + G_umklap
            #aq = linalg.norm(b)
            t0 = timer()
            if s_Irredk:
                jrk = kqm.kii_ind[jk]
                # And next create alpha, beta, gamma for k+q
                Aeigq = array( conj( ks.all_As[jrk][nbs:nbe,:] ), dtype=complex)  # eigenvector from vector file
                if not DMFT1:
                    if kqm.k_ind[jrk] != jk:                            # the k-q-point is reducible, eigenvector needs additional phase
                        Aeigq *= exp( -2*pi*1j * ks.phase_arg[jk][:] )  # adding phase e^{-i*tau[isym]*(k_irr+G_irr)}
                    alm,blm,clm = lapwc.gap2_set_lapwcoef(kjl, ks.indgk[jk], 2, True, ks.nv[jrk], pw.gindex, radf.abcelo[isp], strc.rmt, strc.vpos, strc.mult, radf.umt[isp], strc.rotloc, latgen.trotij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
                else:
                    kjrr = array(kqm.kirlist[jrk,:])/float(kqm.LCM) 
                    isym = kqm.iksym[jk]
                    timat_ik, tau_ik = strc.timat[isym].T, strc.tau[isym,:]
                    alm,blm,clm = lapwc.dmft1_set_lapwcoef(False, 2, True, kjl, kjrr, timat_ik, tau_ik, ks.indgkir[jrk], ks.nv[jrk], pw.gindex, radf.abcelo[0], strc.rmt, strc.vpos, strc.mult, radf.umt[0], strc.rotloc, latgen.rotij, latgen.tauij, latgen.Vol,  kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
            else:
                Aeigq = array( conj( ks.all_As[jk][nbs:nbe,:] ), dtype=complex)  # eigenvector from vector file
                alm,blm,clm = lapwc.gap2_set_lapwcoef(kjl, ks.indgkir[jk], 2, True, ks.nv[jk], pw.gindex, radf.abcelo[isp], strc.rmt, strc.vpos, strc.mult, radf.umt[isp], strc.rotloc, latgen.trotij, latgen.Vol, kqm.k2cartes, in1.nLO_at, in1.nlo, in1.lapw, in1.nlomax)
            
            (ngj,nLOmax,ntnt,ndf) = shape(clm)
            (ngj,ntnt,ndf) = shape(alm)
            (nbmax,ngj2) = shape(Aeigq)

            # And now change alm,blm,clm to band basis, which we call alfa,beta,gama
            alfp = reshape( la_matmul(Aeigq, reshape(alm, (ngj,ntnt*ndf)) ), (nbmax,ntnt,ndf) )
            betp = reshape( la_matmul(Aeigq, reshape(blm, (ngj,ntnt*ndf)) ), (nbmax,ntnt,ndf) )
            if in1.nlomax > 0:
                gamp = reshape( la_matmul(Aeigq, reshape(clm, (ngj,ntnt*ndf*nLOmax)) ), (nbmax,nLOmax,ntnt,ndf) )
            else:
                gamp = zeros((1,1,1,1),dtype=complex,order='F') # can not have zero size
            
            if debug:
                print('alfp,betp,gamp=', file=fout)
                for ie in range(shape(alfp)[0]):
                    for lm in range(shape(alfp)[1]):
                        print('ie=%3d lm=%3d alfp=%14.10f%14.10f betp=%14.10f%14.10f' % (ie+1,lm+1,alfp[ie,lm,0].real, alfp[ie,lm,0].imag, betp[ie,lm,0].real, betp[ie,lm,0].imag), file=fout)
                        
            t1 = timer()
            #t_times[0] += t1-t0
            
            ### The muffin-tin part : mmat[ie1,ie2,im] = < u^{product}_{im,lb} | psi^*_{ie2,k-q} psi_{ie1,k} > e^{-iq.r_atom} where ie1=[nst,nend] and ie2=[mst,mend]
            # mmat_mt[ie1,ie2,im] = <j_l(|b|r)|psi^*_{ie2,k+b} psi_{ie1,k} > e^{i*b.r_atom}
            big_L = big_l.T
            mmat_mt = fvxcn.calc_minm_mt(-bl,0,nbmax,0,nbmax, alfa,beta,gama,alfp,betp,gamp, s3r, strc.vpos,strc.mult, nmix,big_L, in1.nLO_at,ncore,cgcoef,in1.lmax,loctmatsize)
            
            idf = -1
            imm=0
            psi_psi = zeros((nbmax,nbmax), dtype=complex)
            for iat in range(strc.nat):
              for ieq in range(strc.mult[iat]):
                  idf += 1
                  for irm in range(nmix[iat]):
                      lb = big_l[iat,irm] #
                      dimm = 2*lb+1       # lme-lms
                      if big_d[iat,irm]==s_idistance[iik,idk]:
                          lms, lme = lb**2, (lb+1)**2
                          # Now adding 4*pi(-i)^l * Y_lm(-b) so that we have : psi_psi = 4*pi(-i)^l * Y_lm(-b) <j_l(|b|r)Y_lm(r)|psi^*_{ie2,k+b} psi_{ie1,k} > e^{i*b.r_atom}
                          # Note that 4*pi*(-i)^l Y_lm(-b) Y_lm(r)^* = 4*pi*i^l*Y_lm(b) * Y^*_lm(r) = e^{i*b*r}
                          # which is psi_psi = < e^{-i*b*(r+r_atom)} | psi^*_{ie2,k+b} psi_{ie1,k} >
                          psi_psi += dot(mmat_mt[:,:,imm:imm+dimm], Ylmb[idf,idk,lms:lme])
                      imm += dimm
            
            if DMFT1 and s_Irredk and kqm.k_ind[jrk] != jk:                             # the k-q-point is reducible, eigenvector needs additional phase
                Aeigq *= exp( -2*pi*1j * ks.phase_arg[jk][:] )
                      
            ## The interstitial part : mmat[ie1,ie2,im]*sqrt(Vol) = < 1 |psi^*_{ie2} psi_{ie1} > = < e^{i*(G_{im}=0)*r}|psi_{ie2,k+q}^* |psi_{ie1,k}>_{Interstitials}
            ## mmat[ie1,ie2] = Aeigq[ie2,nvj]^*  mpwipw(nvj,nvi)  Aeigk.T(nvi,ie1)
            i_g0 = pw.convert_ig0_2_array() # Instead of dictionary pw.ig0, which converts integer-G_vector into index in fixed basis, we will use fortran compatible integer array
            indggq = array(range(len(pw.gindex)),dtype=int)
            if s_Irredk:
                nvi, nvj = ks.nv[irk], ks.nv[jrk]
                mmat_it = fvxcn.calc_minm_is(0,nbmax,0,nbmax,Aeigk,Aeigq,G_umklap,mpwipw,nvi,nvj,ks.indgk[ik],ks.indgk[jk],indggq,pw.gindex,i_g0,latgen.Vol)
            else:
                nvi, nvj = ks.nv[ik], ks.nv[jk]
                mmat_it = fvxcn.calc_minm_is(0,nbmax,0,nbmax,Aeigk,Aeigq,G_umklap,mpwipw,nvi,nvj,ks.indgkir[ik],ks.indgkir[jk],indggq,pw.gindex,i_g0,latgen.Vol)
            ## Combine MT and interstitial part together
            psi_psi1 = mmat_it[:,:,0]*sqrt(latgen.Vol)
            psi_psi3 = psi_psi + psi_psi1

            # M_{m,n}   = < psi_{m,k} | e^{-i b r} |psi_{n,k+b}>
            M_m_n = psi_psi3.conj()   # M_{ie1,ie2}=<psi_{ie2,k+q}|psi_{ie1,k}>^*
            #print(ik+1,jk+1, '%4d'*3 % tuple(pair_umklap[ik,idk,:]), file=fo_mmn)
            #for n in range(len(M_m_n)):
            #    for m in range(len(M_m_n)):
            #        print('%18.14f %18.14f' % (M_m_n[m,n].real, M_m_n[m,n].imag), file=fo_mmn)
            
            M_all[iik-ikstart,idk,:,:] = M_m_n
            #M_c[idk,:,:] = M_m_n
            dst=s_distances[s_idistance[iik,idk]]
            if s_Irredk:
                print('ik=', ik, 'jk=', jk, 'dst=%.5f'%(dst,), 'irk=', irk, 'jrk=', jrk, 'iik=', kqm.k_ind[irk],'!=', ik, 'jjk=', kqm.k_ind[jrk],'!=', jk, 'b=', bl, 'G_umklap=', G_umklap, '<psi|psi>=', file=fout)
            else:
                print('ik=', ik, 'jk=', jk, 'dst=%.5f'%(dst,), 'b=', bl, 'G_umklap=', G_umklap, '<psi|psi>=', file=fout)
            pg3.PrintMC(M_m_n,file=fout)
            if debug:
                print(file=fout)
                pg3.PrintMC(psi_psi,file=fout)
                print(file=fout)
                pg3.PrintMC(psi_psi1,file=fout)
            
        #dR=zeros((3,nbe-nbs))
        #ds=nbs-ibs
        ## bands are arranged in [ibs,nbs,nbe,ibe]
        #for imu in range(3):  # over x,y,z in lattice coordinates
        #    for i in range(nbs-ibs):
        #        dR[imu,0] += 0.5*(1.0-sum(abs(M_c[  2*imu,:,i])**2)) # 1-\sum_m |<psi_{m,k}|psi_{i,k+e_{imu}}>|^2
        #        dR[imu,0] += 0.5*(1.0-sum(abs(M_c[2*imu+1,:,i])**2)) # 1-\sum_m |<psi_{m,k}|psi_{i,k-e_{imu}}>|^2
        #    for i in range(nbs-ibs,nbe-ibs):
        #        dR[imu,i-ds] += 0.5*(1.0-sum(abs(M_c[  2*imu,:,i])**2)) # 1-\sum_m |<psi_{m,k}|psi_{i,k+e_{imu}}>|^2
        #        dR[imu,i-ds] += 0.5*(1.0-sum(abs(M_c[2*imu+1,:,i])**2)) # 1-\sum_m |<psi_{m,k}|psi_{i,k-e_{imu}}>|^2
        #    for i in range(nbe-ibs,ibe-ibs):
        #        dR[imu,-1] += 0.5*(1.0-sum(abs(M_c[  2*imu,:,i])**2)) # 1-\sum_m |<psi_{m,k}|psi_{i,k+e_{imu}}>|^2
        #        dR[imu,-1] += 0.5*(1.0-sum(abs(M_c[2*imu+1,:,i])**2)) # 1-\sum_m |<psi_{m,k}|psi_{i,k-e_{imu}}>|^2
        #dR[:,0]  /= (nbs-ibs+1)
        #dR[:,-1] /= (ibe-nbe+1)
        #for imu in range(3):
        #    # average over (|k+e_{imu}-k|^2 , |k-e_{imu}-k|^2)
        #    dst=0.5*(s_distances[s_idistance[iik,2*imu]]+s_distances[s_idistance[iik,2*imu+1]])
        #    dR[imu,:] /= dst**2
        #print('bands=', [ibs,nbs,nbe,ibe],file=fout)
        #for i in range(0,nbe-nbs): #  over all the relevant bands, index i
        #    print('R[ik='+str(ik)+',i='+str(nbs+i)+']=', dR[:,i].tolist(),file=fout)
        #R[iik,:,:]=dR

    #M_all=zeros((ikend-ikstart,nntot,nbe-nbs,nbe-nbs),dtype=complex)        
    if Parallel and msize>1:
        if mrank==master:
            M_gather=zeros((nnkp,nntot,nbe-nbs,nbe-nbs),dtype=complex)
        else:
            M_gather=None
        dsize = nntot*(nbe-nbs)**2*2
        comm.Barrier()
        comm.Gatherv(M_all,[M_gather,sendcounts*dsize,displacements*dsize,MPI.DOUBLE], root=master)
        if mrank==master:
            M_all = M_gather
        
    if mrank==master:
        save('M_all.npy', M_all)
        
    # M_{m,n}   = < psi_{m,k} | e^{-i b r} |psi_{n,k+b}>
    # M_all[k,idir,m,n] ~ <u_m(k)|u_n(k+e_{idir}*small)>
    #return M_all
    
def Save_Eigenvalues(case, ks_Ebnd_isp, nbs, nbe):
    fo_eig = open(case+'.eig', 'w')
    for ik in range(len(ks_Ebnd_isp)):
        for i,ib in enumerate(range(nbs,nbe)):
            print(i+1,ik+1, ks_Ebnd_isp[ik,ib]*H2eV, file=fo_eig)
            #print(i+1,ik+1, ks_Ebnd_isp[irk,ib], file=fo_eig)
    fo_eig.close()


def Shift_to_1BZ(k,ndiv):
    " This gives k_{1BZ}-k"
    g = zeros(3,dtype=int)
    for i in range(3):
        if k[i]>0:
            g[i]=-int(floor(k[i]/ndiv[i]))
        elif k[i]<0:
            g[i]=int(ceil(-k[i]/ndiv[i]))
    return g


def Find_nbs_nbe(Ebnd, nbs, nbe, pairs, fout):
    nbs_nbe=[]
    for iik,ik in enumerate(pairs):
        nbs_nbe.append( pg3.Find_nbs_nbei(ik, Ebnd, pairs[ik], nbs, nbe, fout) )
    return nbs_nbe

def Compute_mmn(nbs,nbe):
    nspin = 1
    kmr=1  # if we want to increase the G-mesh as compared to case.in1 RKmax
    pwm=2  # plane wave mesh parameter
    
    (case,fout) = pg3.Initialize('a')

    npair = loadtxt(case+'.nnkp')
    npair = array(npair,dtype=int)
    ii=0
    pairs={0:[]}
    for pair in npair:
        if pair[0]!=ii:
            pairs[pair[0]]=[pair[1]]
            ii = pair[0]
        else:
            pairs[pair[0]].append(pair[1])
    #print(pairs)
    
    strc = w2k.Struct(case, fout)
    latgen = w2k.Latgen(strc, fout)
    #latgen.Symoper(strc, fout) #latgen.Symoper(strc, fout) I think symope is called in Latgen initialization
    #kqm = KQmesh(nkdivs, k0shift, strc, latgen, fout)
    
    ks = KohnShamSystem(case, strc, nspin, fout)
    in1 = w2k.In1File(case, strc, fout)#, io.lomax)
    ks.Set_nv(in1.nlo_tot)
    

    fklist = open(case+'.klist', 'r')
    line = fklist.readline()
    mt = re.search('div:\s*\(\s*(\d+)\s*(\d+)\s*(\d+)\)', line)
    nkdivs = [int(mt.group(1)),int(mt.group(2)),int(mt.group(3))]
    k0shift=[0,0,0]
    if mrank==master: print('kdivs=', nkdivs, file=fout)
    
    kqm = KQmesh(nkdivs, k0shift, strc, latgen, fout)
    pw = PlaneWaves(ks.hsrws, kmr, pwm, case, strc, in1, latgen, kqm, False, fout)
    
    nsp, nkp, band_max = shape(ks.Ebnd)
    print('nsp=', nsp, 'nkp=', nkp, 'band_max=', band_max, file=fout)
    print('len(klist)=', len(ks.klist), file=fout)
    ks.PRINT = False
    ks.VectorFileRead(case, strc, latgen, kqm, pw, fout, in1, HaveFullBZ=False)
    
    (Elapw, Elo) = w2k.get_linearization_energies(case, in1, strc, nspin, fout)
    in1.Add_Linearization_Energy(Elapw, Elo)
    Vr = w2k.Read_Radial_Potential(case, strc.nat, nspin, strc.nrpt, fout)
    radf = w2k.RadialFunctions(in1,strc,ks.Elapw,ks.Elo,Vr,nspin,fout)
    del Vr
    radf.get_ABC(in1, strc, fout)

    
    EF=Find_EF(case)
    print('EF=', EF, file=fout)
    ks.Ebnd -= EF*Ry2H
    
    Irredk = False 
    if mrank==master:
        for ib in range(shape(ks.Ebnd[0])[1]):
            Emin = min(ks.Ebnd[0][:,ib])*H2eV
            Emax = max(ks.Ebnd[0][:,ib])*H2eV
            print('band i=%3d has range %12.6f : %12.6f eV' %(ib,Emin,Emax), file=fout)
            if (nbe<=nbs):
                print('band i=%3d has range %12.6f : %12.6f eV' %(ib,Emin,Emax))
        if (nbe<=nbs):
            print('For which bands do you want to compute Quantum geometric tensor? Give the answer in   istart:iend')
            band_range = input('For which bands do you want to compute Quantum geometric tensor?  istart:iend  ')
            mm = re.search('(\d+)\s*:(\d+)', band_range)
            nbs,nbe = int(mm.group(1)),int(mm.group(2))+1
            print(str(nbs)+'<= i_band <'+str(nbe))
        Save_Eigenvalues(case, ks.Ebnd[0], nbs, nbe)
        
    if Parallel:
        comm.Barrier()
        (nbs,nbe) = comm.bcast((nbs,nbe), root=master)

    nbs_nbe = Find_nbs_nbe(ks.Ebnd[0], nbs, nbe, pairs, fout)
    
    nntot = max([len(p) for p in pairs.values()])
    idistance = zeros((len(pairs),nntot),dtype=int)
    distances=[]
    if mrank==master:    
        print('pairs of k-points at which we are going to compute M(k,b)_{m,n}= <psi_{m,k}|e^{-i*b*r}|psi_{n,k+b}>',file=fout)
    
    for iik,ik in enumerate(pairs):
        for il in range(len(pairs[ik])):
            #print('%4d %4d' % (ik, pair[ik,il]), '  %2d %2d %2d'%tuple(pair_umklap[ik,il,:]))
            jk = pairs[ik][il]
            k1 = array(ks.klist[ik])
            k2 = array(ks.klist[jk])
            aq = linalg.norm(kqm.k2cartes @ (k1-k2))
            ip = where( abs(array(distances)-aq)<1e-7 )[0]
            if len(ip):
                idistance[iik,il] = ip[0]
            else:
                idistance[iik,il] = len(distances)
                distances.append(aq)
            if mrank==master:
                print('%4d %4d' % (ik, jk),' %.7f'%(aq,),' k['+str(ik)+']=',ks.klist[ik],'k['+str(jk)+']=', ks.klist[jk], file=fout)
    print('distances=', distances, file=fout)
    
    # M_{m,n}   = < psi_{m,k} | e^{-i b r} |psi_{n,k+b}>
    tnbs = min([nb[0] for nb in nbs_nbe])
    tnbe = max([nb[1] for nb in nbs_nbe])
    if mrank==master:
        print('Final nbs=', tnbs, 'nbe=', tnbe, file=fout)
        print(str(tnbs)+'<= i_band <'+str(tnbe))
    
    Compute_and_Save_BandOverlap_Mmn(pairs,None,distances,idistance,tnbs,tnbe,Irredk,case,strc,latgen,ks,kqm, pw, in1, radf, fout, DMFT1=False, debug=False)
    
    if mrank==master:
        save('klist.npy', ks.klist)
        save('Ebnd.npy', ks.Ebnd[0])
        with open('pairs.pkl', 'wb') as f:
            pickle.dump(pairs, f)
            pickle.dump([nbs,nbe,tnbs,tnbe,latgen.Vol], f)
            pickle.dump(distances, f)
            pickle.dump(idistance, f)
    return

def Compute_amn():
    #klist = load('klist.npy')
    #Ebnd = load('Ebnd.npy')
    with open('pairs.pkl', 'rb') as f:
        pairs = pickle.load(f)
        [nbs,nbe,tnbs,tnbe,latgen_Vol] = pickle.load(f)
        distances = pickle.load(f)
        idistance = pickle.load(f)
    nbs0,nbe0=nbs,nbe
    (case,fout) = pg3.Initialize('a')
    #print('shape(klist)=', shape(klist))
    print('At reading: nbs,nbe,tnbs,tnbe=', nbs,nbe,tnbs,tnbe, file=fout)
    print('At reading: nbs,nbe,tnbs,tnbe=', nbs,nbe,tnbs,tnbe)
    
    nspin = 1
    kmr=1  # if we want to increase the G-mesh as compared to case.in1 RKmax
    pwm=2  # plane wave mesh parameter
    
    strc = w2k.Struct(case, fout)
    latgen = w2k.Latgen(strc, fout)
    in1 = w2k.In1File(case, strc, fout)
    
    orbits = Orbitals(strc, in1)
    orbits.ReadIndmflFile(case)
    
    orbts, CFx = orbits.FromIndmfl()

    #print(orbits.Convert_iorb2text(orbts))
    
    rotloc = copy(strc.rotloc)
    for iat in range(len(strc.mult)):
        idf0 = sum(strc.mult[:iat])
        rotloci = rotloc[iat]
        rotlocn = None
        for ieq in range(strc.mult[iat]):
            idf = idf0+ieq
            if idf in orbits.rotloc:
                #print('str.rotloc=', rotloci, 'indmfl.rotloc=', orbits.rotloc[idf])
                if rotlocn is None:
                    rotlocn = orbits.rotloc[idf]
                else:
                    if not allclose(orbits.rotloc[idf],rotlocn):
                        print('WARNING equivalent atoms between ', idf0, 'and', idf0+strc.mult[iat]-1, 'have different rotloc:', file=fout)
                        print(rotlocn, 'and', orbits.rotloc[idf], file=fout)
        if rotlocn is not None:
            rotloc[iat] = rotlocn
    dirc=['x','y','z']
    print('rotloc from case.indmfl file for all atom types:', file=fout)
    for iat in range(len(strc.mult)):
        nam = strc.aname[iat].split()[0] # If space appears in struct file, such name is not working in wannier90, so remove anything after space.
        print(nam+':', file=fout)
        for i in range(3):
            print(rotloc[iat,i,:].tolist(), '# new '+dirc[i]+' axis', file=fout)
        
    latgen.Symoper(strc, fout)
    #kqm = KQmesh(nkdivs, k0shift, strc, latgen, fout)

    #print('latgen.br2=', latgen.br2)
    
    
    ks = KohnShamSystem(case, strc, nspin, fout)
    ks.Set_nv(in1.nlo_tot)
    
    fklist = open(case+'.klist', 'r')
    line = fklist.readline()
    mt = re.search('div:\s*\(\s*(\d+)\s*(\d+)\s*(\d+)\)', line)
    nkdivs = [int(mt.group(1)),int(mt.group(2)),int(mt.group(3))]
    k0shift=[0,0,0]
    if mrank==master: print('kdivs=', nkdivs, file=fout)
    
    kqm = KQmesh(nkdivs, k0shift, strc, latgen, fout)
    #print('kqm.k2icartes=', kqm.k2icartes)
    pw = PlaneWaves(ks.hsrws, kmr, pwm, case, strc, in1, latgen, kqm, False, fout)
    
    nsp, nkp, band_max = shape(ks.Ebnd)
    print('nsp=', nsp, 'nkp=', nkp, 'band_max=', band_max, file=fout)
    print('len(klist)=', len(ks.klist), file=fout)
    ks.PRINT = False
    ks.VectorFileRead(case, strc, latgen, kqm, pw, fout, in1, HaveFullBZ=False)
    
    (Elapw, Elo) = w2k.get_linearization_energies(case, in1, strc, nspin, fout)
    in1.Add_Linearization_Energy(Elapw, Elo)
    Vr = w2k.Read_Radial_Potential(case, strc.nat, nspin, strc.nrpt, fout)
    radf = w2k.RadialFunctions(in1,strc,ks.Elapw,ks.Elo,Vr,nspin,fout)
    del Vr
    radf.get_ABC(in1, strc, fout)

    
    EF=Find_EF(case)
    print('EF=', EF, file=fout)
    ks.Ebnd -= EF*Ry2H
    
    Irredk = False 
    
    k_ind = array(arange(int(len(ks.klist)/7))*7,dtype=int) # k-points in original case.klist_band
    
    #for iik,ik in enumerate(k_ind):
    #    print(iik, ik, ks.klist[ik])
    
    #(orbs,orb_names,orb_w) = orbits.FindRelevantOrbitals(tnbs,tnbe,k_ind,rotloc, strc, latgen, ks, kqm, pw, in1, radf, fout, debug=False)
    #for i,ib in enumerate(orbs):
    #    print(i,ib, orb_names[i], orb_w[i])

    #print('orbts=', orbts)
    #print('CFx=', CFx)
    
    Compute_and_Save_Projection_amn(tnbs,tnbe, rotloc, k_ind, case, orbits, orbts, CFx, strc, latgen, ks, kqm, pw, in1, radf, fout, debug=False)
    
    
    sys.exit(0)
    
    #Compute_and_Save_BandOverlap_Mmn(pairs,None,distances,idistance,tnbs,tnbe,Irredk,case,strc,latgen,ks,kqm, pw, in1, radf, fout, DMFT1=False, debug=False)
    
    #if mrank==master:
    #    save('klist.npy', ks.klist)
    #    save('Ebnd.npy', ks.Ebnd[0])
    #    with open('pairs.pkl', 'wb') as f:
    #        pickle.dump(pairs, f)
    #        pickle.dump([nbs,nbe,tnbs,tnbe,latgen.Vol], f)
    #        pickle.dump(distances, f)
    #        pickle.dump(idistance, f)
    return
    
#def Compute_QGT(nbs_nbe=None):
#    # M[k,b]_{m,n}   = < psi_{m,k} | e^{-i b r} |psi_{n,k+b}>
#    M_all = load('M_all.npy')
#    klist = load('klist.npy')
#    Ebnd = load('Ebnd.npy')
#    with open('pairs.pkl', 'rb') as f:
#        pairs = pickle.load(f)
#        [nbs,nbe,tnbs,tnbe,latgen_Vol] = pickle.load(f)
#        distances = pickle.load(f)
#        idistance = pickle.load(f)
#    nbs0,nbe0=nbs,nbe
#    (case,fout) = Initialize('a')
#    print('shape(klist)=', shape(klist))
#    print('shape(M_all)=', shape(M_all))
#    print('At reading: nbs,nbe,tnbs,tnbe=', nbs,nbe,tnbs,tnbe, file=fout)
#    print('At reading: nbs,nbe,tnbs,tnbe=', nbs,nbe,tnbs,tnbe)
#    #print('pairs=', pairs)
#    #print('distances=', distances)
#    #print('idistance=', idistance)
#    strc = w2k.Struct(case, fout)
#    latgen = w2k.Latgen(strc, fout)
#    latgen.Symoper(strc, fout)
#    # What W2k uses in stored klist. It is K2icartes@(i/N1,j/N2,k/N3)
#    aaa = array([strc.a, strc.b, strc.c])
#    if latgen.ortho or strc.lattice[1:3]=='CXZ':
#        k2icartes = array((aaa/(2*pi)*latgen.br2).round(),dtype=int)
#        k2cartes = 2*pi/aaa*identity(3) # == latgen.pia
#        # When apply k2cartes . k2icartes . (i,j,k) we get
#        # 2*pi/aaa[:] . BR2 . aaa[:]/(2*pi) (i,j,k) = 1/aaa[:] . BR2 . aaa[:]
#    else:
#        k2icartes = identity(3,dtype=int)
#        k2cartes  = latgen.br2
#
#    if nbs_nbe==None:
#        nbs_nbe = [(i-nbs0,i+1-nbs0) for i in range(nbs,nbe)]
#    
#    direction=['x','-x','y','-y','z','-z']
#    QGM = []
#    for iik,ik in enumerate(pairs):
#        dst = zeros(3)
#        for imu in range(3):
#            # average over (|k+e_{imu}-k|^2 , |k-e_{imu}-k|^2)
#            dst[imu]=0.5*(distances[idistance[iik,2*imu]]+distances[idistance[iik,2*imu+1]])
#        tQGM=[]
#        print('iik=', iik, 'ik=', ik, 'dst=%7.5f %7.5f %7.5f'%tuple(dst), klist[ik].tolist(), file=fout)
#        for nbs,nbe in nbs_nbe:
#            ibs,ibe = Find_nbs_nbei(ik, Ebnd, pairs[ik], nbs+nbs0, nbe+nbs0, fout)
#            ibs-=nbs0
#            ibe-=nbs0
#            _ibs_ = ibs+nbs0-tnbs
#            _ibe_ = ibe+nbs0-tnbs
#            M_c = M_all[iik,:,_ibs_:_ibe_,_ibs_:_ibe_]
#            print('  bands=',[ibs,nbs,nbe,ibe],[_ibs_,nbs+nbs0-tnbs,nbe+nbs0-tnbs,_ibe_],
#                      ([i for i in range(nbs-ibs)],[i for i in range(nbs-ibs,nbe-ibs)],[i for i in range(nbe-ibs,ibe-ibs)]),
#                      'M_mmn=', 'shape(M_all)=', shape(M_all[iik]), 'shape(M_c)=', shape(M_c), file=fout)
#            #print(shape(M_c), file=fout)
#            for imu in range(shape(M_all)[1]):
#                print(' M_'+str(direction[imu]), file=fout)
#                PrintMC(M_c[imu,:,:],file=fout)
#            tR=zeros((nbe-nbs,3))
#            ds=nbs-ibs
#            # bands are arranged in [ibs,nbs,nbe,ibe]
#            for imu in range(3):  # over x,y,z in lattice coordinates
#                for i in range(nbs-ibs):
#                    tR[0,imu] += 0.5*(1.0-sum(abs(M_c[  2*imu,:,i])**2)) # 1-\sum_m |<psi_{m,k}|psi_{i,k+e_{imu}}>|^2
#                    tR[0,imu] += 0.5*(1.0-sum(abs(M_c[2*imu+1,:,i])**2)) # 1-\sum_m |<psi_{m,k}|psi_{i,k-e_{imu}}>|^2
#                for i in range(nbs-ibs,nbe-ibs):
#                    tR[i-ds,imu] += 0.5*(1.0-sum(abs(M_c[  2*imu,:,i])**2)) # 1-\sum_m |<psi_{m,k}|psi_{i,k+e_{imu}}>|^2
#                    tR[i-ds,imu] += 0.5*(1.0-sum(abs(M_c[2*imu+1,:,i])**2)) # 1-\sum_m |<psi_{m,k}|psi_{i,k-e_{imu}}>|^2
#                for i in range(nbe-ibs,ibe-ibs):
#                    tR[-1,imu] += 0.5*(1.0-sum(abs(M_c[  2*imu,:,i])**2)) # 1-\sum_m |<psi_{m,k}|psi_{i,k+e_{imu}}>|^2
#                    tR[-1,imu] += 0.5*(1.0-sum(abs(M_c[2*imu+1,:,i])**2)) # 1-\sum_m |<psi_{m,k}|psi_{i,k-e_{imu}}>|^2
#            tR[0 ,:]  /= (nbs-ibs+1)
#            tR[-1,:] /= (ibe-nbe+1)
#            for imu in range(3):
#                #tR[:,imu] *= (8*pi**2/latgen_Vol)/dst[imu]**2
#                tR[:,imu] *= 2.0/dst[imu]**2
#            #print('bands=', [ibs,nbs,nbe,ibe],file=fout)
#            for i in range(0,nbe-nbs): #  over all the relevant bands, index i
#                print('R[ik='+str(iik)+',i='+str(nbs+i)+']=', tR[i,:].tolist(),file=fout)
#            tQGM.append(tR)
#        QGM.append(tQGM)
#
#    savetxt('nbs_nbe.dat', array(nbs_nbe,dtype=int))
#    fq = open('QGT.dat','w')
#    fe = open('eQGT.dat','w')
#    dire=['x','y','z']
#    print('# %2s %8s' % ('ik', '|d|'), end='',file=fq)
#    for ib in range(len(nbs_nbe)):
#        for i in range(nbs_nbe[ib][0],nbs_nbe[ib][1]):
#            for imu in range(3):
#                print('%12s '%('band['+str(i)+']:'+dire[imu],), end='', file=fq)
#    print(file=fq)
#                
#    kp = klist[0]
#    dst=0
#    for iik,ik in enumerate(pairs):
#        kn = array(klist[ik])
#        dst += linalg.norm(k2cartes @ (kn-kp))
#        print('%3d %8.5f' % (iik, dst), end='',file=fq)
#        print('%10.5f' % (dst,), end='',file=fe)
#        for ib,(nbs,nbe) in enumerate(nbs_nbe):
#            for i in range(nbs,nbe):
#                print('%12.5f'%(Ebnd[ik,i+nbs0]*H2eV,), end='', file=fe)
#                for imu in range(3):
#                    print('%12.4f ' % QGM[iik][ib][i-nbs,imu],  end='',file=fq)
#        print(file=fe)
#        print(file=fq)
#        kp = kn
#    print(file=fq)
#    

def Back_Rename_klist():
    if mrank==master:
        klist_orig_names = glob.glob('*.klist*.bak')
        if klist_orig_names:
            name = klist_orig_names[0]
            #print(name, name[:-4])
            os.rename(name, name[:-4])
    
    
if __name__ == '__main__':
    #Compute_amn()
    #sys.exit(0)

    
    Back_Rename_klist()
    Compute_mmn(nbs=0,nbe=0)
    if mrank==master:
        pg3.Compute_QGT()



    
    
    # SrVO3: [(0,9),(9,12),(12,20)]
    # MnO:  [(0,3),(3,8)]
    # FeSe: [(0,6),(6,16),(16,23)]
    # LaRuSi: [(0,10),(10,17),(17,25)]
    # Si: [(0,4),(4,10)]
    # LSCO : [(0,16),(16,17)]
    # Bi2212 : [(0,44),(44,50),(51,55)]
    # CaRh2 : [(0,14),(14,18),(18,25)]
