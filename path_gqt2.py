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

def Check_Irreducible_wedge(ks, kqm, fout):
    if (len(ks.all_As)==len(kqm.kirlist)):
        print('From vector-file we have irreducible wedge k-points only', file=fout)
        return True
    elif (len(ks.all_As)==len(kqm.klist)):
        print('From vector-file we have all k-points (not irreducible wedge)', file=fout)
        return False
    else:
        print('ERROR len(ks.all_As)=', len(ks.all_As), ' is not one of ', len(kqm.kirlist), 'or', len(kqm.klist))
        print('ERROR len(ks.all_As)=', len(ks.all_As), ' is not one of ', len(kqm.kirlist), 'or', len(kqm.klist), file=fout)
        sys.exit(1)


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
    latgen.Symoper(strc, fout)
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
            band_range = input('For which bands do you want to compute Quantum geometric tensor?  istart : iend  ')
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
