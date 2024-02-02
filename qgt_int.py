#!/usr/bin/env python
# @Copyright 2020 Kristjan Haule

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

from cmn import Ry2H,H2eV,Ry2eV,H2Ry,Br2Ang
#from inout import InOut
from kqmesh import KQmesh
import mcommon as mcmn
from kohnsham import KohnShamSystem
from planewaves import PlaneWaves
from matel2band import MatrixElements2Band
from productbasis import ProductBasis
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


def Compute_and_Save_BandOverlap_Mmn(pair, pair_umklap, s_distances, s_idistance, s_nbe, s_nbs, s_Irredk, case, strc, latgen, ks, kqm, pw, in1, radf, fout, DMFT1=False, debug=False):
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

    (nnkp, nntot) = shape(pair)

    # It turns out there are only a few possible angles Y_lm(b), for which Y_lm needs to be evaluates
    # We first find all possible angles of vector b.
    b_angles = []
    b_pair = zeros(shape(pair), dtype=int)
    for ik in range(len(kqm.klist)):
        for i in range(nntot):
            jk = pair[ik,i]
            k1 = kqm.klist[ik,:]/kqm.LCM
            k2 = kqm.klist[jk,:]/kqm.LCM
            b = k2-k1+ dot(kqm.k2icartes,pair_umklap[ik,i,:])
            ab = linalg.norm(b)
            if ab!=0:
                b_ang = b/ab
            else:
                b_ang = b
            ic = FindInArray( b_ang, b_angles)
            if ic<0:
                b_pair[ik,i] = len(b_angles)
                b_angles.append(b_ang)
            else:
                b_pair[ik,i] = ic
    
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

    print('writing: ', case+'.mmn', file=fout)

    #fo_mmn = open(case+'.mmn', 'w')
    #print('# code PyGW:wannier.py', date.today().strftime("%B %d, %Y"), file=fo_mmn)
    #print(s_nbe-s_nbs, len(kqm.klist), nntot, file=fo_mmn)
    
    V_rmt = 0
    for iat in range(strc.nat):
        for ieq in range(strc.mult[iat]):
            V_rmt += 4*pi*strc.rmt[iat]**3/3
    print('V_rmt/V=', V_rmt/latgen.Vol, 'V_I/V=', 1-V_rmt/latgen.Vol, 'V_rmt=', V_rmt, 'V=', latgen.Vol, file=fout)

    ikstart,ikend,sendcounts,displacements = mcmn.mpiSplitArray(mrank, msize, len(kqm.klist) )
    print('processor rank=', mrank, 'with size=', msize, 'will do ikstart=', ikstart, 'ikend=', ikend, 'displacements=',displacements, file=fout)
    fout.flush()

    M_all=zeros((ikend-ikstart,shape(pair)[1],s_nbe-s_nbs,s_nbe-s_nbs),dtype=complex)
    
    idf = -1
    for ik in range(ikstart,ikend):#range(len(kqm.klist)):
        kil = kqm.klist[ik,:]/kqm.LCM  # k1==kil in semi-cartesian form
        irk = kqm.kii_ind[ik]          # the index to the correspodning irreducible k-point
        if s_Irredk:
            # First create alpha, beta, gamma for k
            Aeigk = array(ks.all_As[irk][s_nbs:s_nbe,:], dtype=complex)   # eigenvector from vector file
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
            Aeigk = array(ks.all_As[ik][s_nbs:s_nbe,:], dtype=complex)   # eigenvector from vector file
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

        
        if DMFT1 and kqm.k_ind[irk] != ik:                       # not irreducible
            # For interstitials we need to transform now, because we did not transform the eigenvectors before.
            Aeigk *= exp( 2*pi*1j * ks.phase_arg[ik][:])  # phase for reducible: e^{i*tau[isym]*(k_irr+G_irr)}, where tau[isym] is the part of the group operation from irreducible to reducible k-point
        
        if debug:
            print('alfa,beta,gama=', file=fout)
            for ie in range(shape(alfa)[0]):
                for lm in range(shape(alfa)[1]):
                    print('ie=%3d lm=%3d alfa=%14.10f%14.10f beta=%14.10f%14.10f' % (ie+1,lm+1,alfa[ie,lm,0].real, alfa[ie,lm,0].imag, beta[ie,lm,0].real, beta[ie,lm,0].imag), file=fout)
        
        for idk in range(nntot):
            jk = pair[ik,idk]
            # M_{m,n}   = < psi_{m,k} | e^{-i b r} |psi_{n,k+b}>
            # M_{m,n}^* = < psi_{n,k+b}| e^{i b r} |psi_{m,k}>
            # M_{m,n}^* = < psi_{n,k-q}| e^{-i q r}|psi_{m,k}> where q=-b. The last form is equivalent to our definition of M matrix element in GW.
            # k2 = k1 + b - G_umklap => b = k2-k1+G_umklapp
            kjl = kqm.klist[jk,:]/kqm.LCM  # k2==kjl=k+b in semi-cartesian form
            G_umklap = array(dot(kqm.k2icartes,pair_umklap[ik,idk,:]), dtype=int)
            bl = kjl-kil + G_umklap
            #aq = linalg.norm(b)
            jrk = kqm.kii_ind[jk]
            t0 = timer()
            if s_Irredk:
                # And next create alpha, beta, gamma for k+q
                Aeigq = array( conj( ks.all_As[jrk][s_nbs:s_nbe,:] ), dtype=complex)  # eigenvector from vector file
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
                Aeigq = array( conj( ks.all_As[jk][s_nbs:s_nbe,:] ), dtype=complex)  # eigenvector from vector file
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
                      if big_d[iat,irm]==s_idistance[ik,idk]:
                          lms, lme = lb**2, (lb+1)**2
                          # Now adding 4*pi(-i)^l * Y_lm(-b) so that we have : psi_psi = 4*pi(-i)^l * Y_lm(-b) <j_l(|b|r)Y_lm(r)|psi^*_{ie2,k+b} psi_{ie1,k} > e^{i*b.r_atom}
                          # Note that 4*pi*(-i)^l Y_lm(-b) Y_lm(r)^* = 4*pi*i^l*Y_lm(b) * Y^*_lm(r) = e^{i*b*r}
                          # which is psi_psi = < e^{-i*b*(r+r_atom)} | psi^*_{ie2,k+b} psi_{ie1,k} >
                          psi_psi += dot(mmat_mt[:,:,imm:imm+dimm], Ylmb[idf,idk,lms:lme])
                      imm += dimm
            
            if DMFT1 and kqm.k_ind[jrk] != jk:                             # the k-q-point is reducible, eigenvector needs additional phase
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
            
            
            M_all[ik-ikstart,idk,:,:] = M_m_n
            dst=s_distances[s_idistance[ik,idk]]
            print('ik=', ik, 'jk=', jk, 'dst=%.5f'%(dst,), 'irk=', irk, 'jrk=', jrk, 'iik=', kqm.k_ind[irk],'!=', ik, 'jjk=', kqm.k_ind[jrk],'!=', jk, 'b=', bl, 'G_umklap=', G_umklap, '<psi|psi>=', file=fout)
            PrintMC(M_m_n,file=fout)
            if debug:
                print(file=fout)
                PrintMC(psi_psi,file=fout)
                print(file=fout)
                PrintMC(psi_psi1,file=fout)

    
    if Parallel and msize>1:
        if mrank==master:
            M_gather=zeros((shape(pair)[0],shape(pair)[1],s_nbe-s_nbs,s_nbe-s_nbs),dtype=complex)
        else:
            M_gather=None
        dsize = shape(pair)[1]*(s_nbe-s_nbs)**2 * 2
        #print('mrank=', mrank, 'dsize=', dsize, 'shape(M_all)=', shape(M_all), 'shape(M_gather)=', shape(M_gather), 'sendcounts*dsize=', sendcounts*dsize, 'displacements*dsize=', displacements*dsize)
        #print('len(M_all)=', len(M_all.ravel()))
        #if mrank==master:
        #    print('len(M_gather)=', len(M_gather.ravel()))
        comm.Barrier()
        comm.Gatherv(M_all,[M_gather,sendcounts*dsize,displacements*dsize,MPI.DOUBLE], root=master)
        if mrank==master:
            M_all = M_gather
        
    if mrank==master:
        save('M_mmn.npy', M_all)
                
    # M_{m,n}   = < psi_{m,k} | e^{-i b r} |psi_{n,k+b}>
    #return M_all
    
def Save_Eigenvalues(case, ks_Ebnd_isp, nbs, nbe, Irredk, kqm):
    fo_eig = open(case+'.eig', 'w')
    for ik in range(len(kqm.klist)):
        if Irredk:
            irk = kqm.kii_ind[ik]
        else:
            irk = ik
        for i,ib in enumerate(range(nbs,nbe)):
            print(i+1,ik+1, ks_Ebnd_isp[irk,ib]*H2eV, file=fo_eig)
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

def Initialize(mode='w', Print=True, out_file='QGT.out'):
    if mrank==master:    
        fout = open(out_file, mode)
    else:
        fout = open(os.devnull,"w")
    struct_names = glob.glob('*.struct')
    if len(struct_names)==0:
        print('ERROR : Could not find a *.struct file present')
        sys.exit(0)
    else:
        case = struct_names[0].split('.')[0]
    if Print and mrank==master: print('case=', case)
    fklist = open(case+'.klist', 'r')
    line = fklist.readline()
    mt = re.search('div:\s*\(\s*(\d+)\s*(\d+)\s*(\d+)\)', line)
    nkdivs = [int(mt.group(1)),int(mt.group(2)),int(mt.group(3))]
    k0shift=[0,0,0]
    if Print and mrank==master: print('kdivs=', nkdivs)
    return (case, nkdivs, k0shift, fout)

    
def Compute_mmn(nbs,nbe, cmpEF=True):
    (case, nkdivs, k0shift, fout) = Initialize('w')
    nspin=1
    kmr=1  # if we want to increase the G-mesh as compared to case.in1 RKmax
    pwm=2  # plane wave mesh parameter
    
    strc = w2k.Struct(case, fout)
    latgen = w2k.Latgen(strc, fout)
    latgen.Symoper(strc, fout)
    kqm = KQmesh(nkdivs, k0shift, strc, latgen, fout)

    # creates a list of k-point pairs that we will need to compute QGT
    npair=set()
    for i0 in range(kqm.ndiv[0]):
        for i1 in range(kqm.ndiv[1]):
            for i2 in range(kqm.ndiv[2]):
                pairs=[ ([i0,i1,i2],[i0+1,i1,i2]),
                        ([i0,i1,i2],[i0,i1+1,i2]),
                        ([i0,i1,i2],[i0,i1,i2+1]),
                        ([i0+1,i1,i2],[i0,i1+1,i2]),
                        ([i0+1,i1,i2],[i0,i1,i2+1]),
                        ([i0,i1+1,i2],[i0+1,i1,i2]),
                        ([i0,i1+1,i2],[i0,i1,i2+1]),
                        ([i0,i1,i2+1],[i0+1,i1,i2]),
                        ([i0,i1,i2+1],[i0,i1+1,i2])]
                for k1,k2 in pairs:
                    k1 = array(k1,dtype=int)
                    k2 = array(k2,dtype=int)
                    gs1 = -Shift_to_1BZ(k1,kqm.ndiv)
                    gs2 = -Shift_to_1BZ(k2,kqm.ndiv)
                    k1_bz = k1-gs1*kqm.ndiv
                    k2_bz = k2-gs2*kqm.ndiv
                    ik1 = kqm.k2indx(k1_bz)
                    ik2 = kqm.k2indx(k2_bz)
                    G = gs2-gs1
                    npair.add((ik1,ik2,G[0],G[1],G[2]))
                    #print([i0,i1,i2],'pair=', k1,k2, ik1,ik2,G)
    npair = list(npair) # from set back to list, so that we don't duplicate pairs
    npair.sort(key= lambda x:x[0]) # and now sorted by ik
    nnkp = kqm.ndiv[0]*kqm.ndiv[1]*kqm.ndiv[2] # all k-points in the mesh
    n_negh=zeros(nnkp, dtype=int)
    print('pairs of k-points at which we are going to compute M(k,b)_{m,n}= <psi_{m,k}|e^{-i*b*r}|psi_{n,k+b}>',file=fout)
    for ipair in npair:
        n_negh[ipair[0]] += 1
        print('%4d %4d' % (ipair[0], ipair[1]), '  %2d %2d %2d'%tuple(ipair[2:5]), file=fout)
    print(file=fout)
    
    nntot=max(n_negh)
    if (min(n_negh)!=nntot):
        print('WARNING : some k-points seem to have different numers of needed neighbors!')
        print('WARNING : some k-points seem to have different numers of needed neighbors!', file=fout)
    
    pair = zeros((nnkp,nntot),dtype=int)
    pair_umklap = zeros((nnkp,nntot,3),dtype=int)
    decode=[{} for ik in range(nnkp)]
    il,ik_previous=0,0
    for ipair in npair:
        if ipair[0]!=ik_previous:
            il=0
        pair[ipair[0],il] = ipair[1]
        pair_umklap[ipair[0],il,:] = ipair[2:5]
        if ipair[1] in decode[ipair[0]]:
            print('ERROR variable decode assumes there is only one needed matrix element between k-points ',ipair[0],',',ipair[1])
            print('We have G_old=', pair_umklap[ipair[0],decode[ipair[0]],:], 'and G_new=', ipair[2:5])
        else:
            decode[ipair[0]][ipair[1]]=il
        ik_previous=ipair[0]
        il+=1
    #for ik in range(len(pair)):
    #    for il in range(shape(pair)[1]):
    #        print('%4d %4d' % (ik, pair[ik,il]), '  %2d %2d %2d'%tuple(pair_umklap[ik,il,:]))
    
    ks = KohnShamSystem(case, strc, nspin, fout)
    in1 = w2k.In1File(case, strc, fout)#, io.lomax)
    ks.Set_nv(in1.nlo_tot)
    
    pw = PlaneWaves(ks.hsrws, kmr, pwm, case, strc, in1, latgen, kqm, False, fout)
    ks.VectorFileRead(case, strc, latgen, kqm, pw, fout, in1)
    (Elapw, Elo) = w2k.get_linearization_energies(case, in1, strc, nspin, fout)
    in1.Add_Linearization_Energy(Elapw, Elo)
    Vr = w2k.Read_Radial_Potential(case, strc.nat, nspin, strc.nrpt, fout)
    radf = w2k.RadialFunctions(in1,strc,ks.Elapw,ks.Elo,Vr,nspin,fout)
    del Vr
    radf.get_ABC(in1, strc, fout)

    ######
    ### DEBUG
    #print(len(kqm.kirlist), len(ks.klist))
    #for ik in range(len(kqm.kirlist)):
    #    k1 = kqm.kirlist[ik,:]/kqm.LCM
    #    k2 = ks.klist[ik]
    #    print(k1, k2, linalg.norm(k1-k2))
    #sys.exit(0)
    ######
    EF=0
    if cmpEF:
        kqm.tetra(latgen, strc, fout)
        if os.path.isfile(case+'.core'):
            wcore = w2k.CoreStates(case, strc, nspin, fout)
            nval = wcore.nval
        else:
            lines = open(case+'.in2','r').readlines()
            nval = float(lines[1].split()[1])
        
        (EF, Eg, evbm, ecbm, eDos) = mcmn.calc_Fermi(ks.Ebnd[0], kqm.atet, kqm.wtet, nval, ks.nspin)
        ks.Ebnd -= EF
        if Eg >= 0:
            print('\n'+'-'*32+'\nFermi: Insulating, KS E_Fermi[eV]=%-12.6f Gap[eV]=%-12.6f  EVBM[eV]=%-12.6f  ECBM[eV]=%-12.6f' % (EF*H2eV, Eg*H2eV, evbm*H2eV, ecbm*H2eV), file=fout)
        else:
            print('\n'+'-'*32+'\nFermi: Metallic, KS E_Fermi[eV]=%-12.6f  DOS[E_f]=%-12.6f' % (EF*H2eV, eDos), file=fout)

    Irredk = Check_Irreducible_wedge(ks, kqm, fout)

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
        Save_Eigenvalues(case, ks.Ebnd[0], nbs, nbe, Irredk, kqm)

    if Parallel:
        (nbs,nbe) = comm.bcast((nbs,nbe), root=master)
        
    idistance = zeros((nnkp,nntot),dtype=int)
    distances=[]
    print('#%5s %5s'%('k1', 'k2'), '   %9s   %5s  %7s'%('umklap','k2irr','|k2-k1|'), file=fout)
    for ik in range(len(pair)):
        for il in range(shape(pair)[1]):
            #print('%4d %4d' % (ik, pair[ik,il]), '  %2d %2d %2d'%tuple(pair_umklap[ik,il,:]))
            jk = pair[ik,il]
            gs = pair_umklap[ik,il]
            irk = kqm.kii_ind[ik]
            jrk = kqm.kii_ind[jk]
            ##
            k1 = kqm.klist[ik,:]/kqm.LCM   # ik actual form
            k2 = kqm.klist[jk,:]/kqm.LCM   # jk actual form in the 1BZ
            G = dot(kqm.k2icartes, gs)     # umklap G
            dk = k2 + G - k1                  # |k2-k1| is distance between the two k-points
            aq = sqrt(dot(dk,dk))             # distance
            # check if this distance already appeared
            #distances[nntot*ik+i] = aq
            ip = where( abs(array(distances)-aq)<1e-6 )[0]
            if len(ip):
                idistance[ik,il] = ip[0]
            else:
                idistance[ik,il] = len(distances)
                distances.append(aq)
            ###
            ### checking that k1 and k2 actual form is what is expected
            print(' %5d %5d'%(ik+1, jk+1), '   %3d %3d %3d'%tuple(gs), '%5d'%(jrk+1), ' %.5f'%(aq,), file=fout)
            #print('dist['+str(nntot*ik+i)+']=',aq)
    print('distances=', distances, file=fout)
    
    # M_{m,n}   = < psi_{m,k} | e^{-i b r} |psi_{n,k+b}>
    # M_all =
    Compute_and_Save_BandOverlap_Mmn(pair, pair_umklap, distances, idistance, nbe, nbs, Irredk, case, strc, latgen, ks, kqm, pw, in1, radf, fout, DMFT1=False, debug=False)

    if mrank==master:
        #save('M_all.npy', M_all)
        save('pair.npy', pair)
        save('pair_umklap.npy', pair_umklap)
        with open('decode.pkl', 'wb') as f:
            pickle.dump(decode, f)
            pickle.dump([nbs,nbe], f)
    return

    #R = zeros((nbe-nbs,3,3),dtype=complex)
    #for i0 in range(kqm.ndiv[0]):
    #    for i1 in range(kqm.ndiv[1]):
    #        for i2 in range(kqm.ndiv[2]):
    #            cases=[ ([i0,i1,i2],[i0+1,i1,i2]),  # (k,k+e_x)
    #                    ([i0,i1,i2],[i0,i1+1,i2]),  # (k,k+e_y)
    #                    ([i0,i1,i2],[i0,i1,i2+1]),  # (k,k+e_z)
    #                    ([i0+1,i1,i2],[i0,i1+1,i2]),# (k+e_x,k+e_y)
    #                    ([i0+1,i1,i2],[i0,i1,i2+1]),# (k+e_x,k+e_z)
    #                    ([i0,i1+1,i2],[i0+1,i1,i2]),# (k+e_y,k+e_x)
    #                    ([i0,i1+1,i2],[i0,i1,i2+1]),# (k+e_y,k+e_z)
    #                    ([i0,i1,i2+1],[i0+1,i1,i2]),# (k+e_z,k+e_x)
    #                    ([i0,i1,i2+1],[i0,i1+1,i2])]# (k+e_z,k+e_y)
    #            icase=[]
    #            for k1,k2 in cases:
    #                k1 = array(k1,dtype=int)
    #                k2 = array(k2,dtype=int)
    #                gs1 = -Shift_to_1BZ(k1,kqm.ndiv)
    #                gs2 = -Shift_to_1BZ(k2,kqm.ndiv)
    #                k1_bz = k1-gs1*kqm.ndiv
    #                k2_bz = k2-gs2*kqm.ndiv
    #                ik = kqm.k2indx(k1_bz)
    #                jk = kqm.k2indx(k2_bz)
    #                G = gs2-gs1
    #                il = decode[ik][jk]
    #                dg = pair_umklap[ik,il]-G
    #                if sum(abs(dg))>0:
    #                    print('ERROR pair_umklap and G are different p_u=', pair_umklap[ik,il], 'G=', G)
    #                if (jk!=pair[ik,il]):
    #                    print('ERROR jk=', jk, 'but from decode we expect', pair[ik,il])
    #                icase.append([ik,il])
    #            for imu in range(3):  # over x,y,z in lattice coordinates
    #                for i in range(0,nbe-nbs): #  over all the relevant bands, index i
    #                    R[i,imu,imu] += 1.0 - sum(abs(M_all[icase[imu][0],icase[imu][1]][:,i])**2)  # 1-\sum_m |<psi_{m,k}|psi_{i,k+e_{imu}}>|^2
    #            for i in range(0,nbe-nbs):
    #                R[i,0,1] += M_all[icase[3][0],icase[3][1]][i,i] - sum(M_all[icase[0][0],icase[0][1]][:,i].conj()*M_all[icase[1][0],icase[1][1]][:,i]) # <psi_{i,k+e_x}|psi_{i,k+e_y}-\sum_j <psi_{j,k}|psi_{i,k+e_x}>^* * <psi_{j,k}|psi_{i,k+e_y}>
    #                R[i,0,2] += M_all[icase[4][0],icase[4][1]][i,i] - sum(M_all[icase[0][0],icase[0][1]][:,i].conj()*M_all[icase[2][0],icase[2][1]][:,i]) # <psi_{i,k+e_x}|psi_{i,k+e_z}-\sum_j <psi_{j,k}|psi_{i,k+e_x}>^* * <psi_{j,k}|psi_{i,k+e_z}>
    #                R[i,1,0] += M_all[icase[5][0],icase[5][1]][i,i] - sum(M_all[icase[1][0],icase[1][1]][:,i].conj()*M_all[icase[0][0],icase[0][1]][:,i]) # <psi_{i,k+e_y}|psi_{i,k+e_x}-\sum_j <psi_{j,k}|psi_{i,k+e_y}>^* * <psi_{j,k}|psi_{i,k+e_x}>
    #                R[i,1,2] += M_all[icase[6][0],icase[6][1]][i,i] - sum(M_all[icase[1][0],icase[1][1]][:,i].conj()*M_all[icase[2][0],icase[2][1]][:,i]) # <psi_{i,k+e_y}|psi_{i,k+e_z}-\sum_j <psi_{j,k}|psi_{i,k+e_y}>^* * <psi_{j,k}|psi_{i,k+e_z}>
    #                R[i,2,0] += M_all[icase[7][0],icase[7][1]][i,i] - sum(M_all[icase[2][0],icase[2][1]][:,i].conj()*M_all[icase[0][0],icase[0][1]][:,i]) # <psi_{i,k+e_z}|psi_{i,k+e_x}-\sum_j <psi_{j,k}|psi_{i,k+e_z}>^* * <psi_{j,k}|psi_{i,k+e_x}>
    #                R[i,2,1] += M_all[icase[8][0],icase[8][1]][i,i] - sum(M_all[icase[2][0],icase[2][1]][:,i].conj()*M_all[icase[1][0],icase[1][1]][:,i]) # <psi_{i,k+e_z}|psi_{i,k+e_y}-\sum_j <psi_{j,k}|psi_{i,k+e_z}>^* * <psi_{j,k}|psi_{i,k+e_y}>
    #
    #del M_all
    #
    #R *= 1/nnkp
    #for i in range(3):
    #    for j in range(3):
    #        # because 1/(dk_i*dk_j) = N_i * N_j
    #        R[:,i,j] *= kqm.ndiv[i]*kqm.ndiv[j]
    #
    #for i in range(nbe-nbs):
    #    print('R for band i=', i)
    #    PrintMC(R[i])
    #
    #Omega = zeros(shape(R))
    #G = zeros(shape(R))
    #for i in range(nbe-nbs):
    #    Omega[i,:] = ((R[i,:,:]-R[i,:,:].T)*1j).real
    #    G[i,:,:]     = 0.5*(R[i,:,:]+R[i,:,:].T).real*(8*pi**2)/latgen.Vol
    #
    #gij = linalg.inv(dot(latgen.br2.T,latgen.br2))
    #for i in range(3):
    #    for j in range(3):
    #        Omega[:,i,j] *= gij[i,j]
    #        G[:,i,j] *= gij[i,j]
    #
    #for i in range(nbe-nbs):
    #    print('Omega for band i=', i)
    #    PrintM(Omega[i])
    #    
    #for i in range(nbe-nbs):
    #    print('G for band i=', i)
    #    PrintM(G[i])
    #            
    #    
    #sys.exit(0)
    #
    #nntot = 6
    #nnkp = kqm.ndiv[0]*kqm.ndiv[1]*kqm.ndiv[2]
    #pair = zeros((nnkp,nntot),dtype=int)
    #pair_umklap = zeros((nnkp,nntot,3),dtype=int)
    #
    #idistance = zeros((nnkp,nntot),dtype=int)
    #distances=[]
    #print('#%5s %5s'%('k1', 'k2'), '   %9s   %5s  %7s'%('umklap','k2irr','|k2-k1|'))
    #for i0 in range(kqm.ndiv[0]):
    #    for i1 in range(kqm.ndiv[1]):
    #        for i2 in range(kqm.ndiv[2]):
    #            ki = array([i0,i1,i2],dtype=int)
    #            ik = kqm.k2indx(ki)
    #            irk = kqm.kii_ind[ik]
    #            
    #            kis = array([[i0+1,i1,i2],[i0-1,i1,i2],[i0,i1+1,i2],[i0,i1-1,i2],[i0,i1,i2+1],[i0,i1,i2-1]],dtype=int)
    #            gs = array([-Shift_to_1BZ(kis[i],kqm.ndiv) for i in range(len(kis))],dtype=int)
    #            jk = zeros(len(kis),dtype=int)
    #            jk_irr = zeros(len(kis),dtype=int)
    #            for i in range(len(kis)):
    #                kis[i] = kis[i] - gs[i]*kqm.ndiv  # [jk0,jk1,jk2] in 1BZ withouth umklap
    #                jk[i] = kqm.k2indx(kis[i])        # jk index in the list [0,...nkp-1]
    #                jk_irr[i] = kqm.kii_ind[jk[i]]    # index in the list of irreducible k-poits
    #                ##
    #                k1 = kqm.klist[ik   ,:]/kqm.LCM   # ik actual form
    #                k2 = kqm.klist[jk[i],:]/kqm.LCM   # jk actual form in the 1BZ
    #                G = dot(kqm.k2icartes, gs[i])     # umklap G
    #                dk = k2 + G - k1                  # |k2-k1| is distance between the two k-points
    #                aq = sqrt(dot(dk,dk))             # distance
    #                # check if this distance already appeared
    #                #distances[nntot*ik+i] = aq
    #                ip = where( abs(array(distances)-aq)<1e-6 )[0]
    #                if len(ip):
    #                    idistance[ik,i] = ip[0]
    #                else:
    #                    idistance[ik,i] = len(distances)
    #                    distances.append(aq)
    #                ###
    #                pair[ik,i] = jk[i]
    #                pair_umklap[ik,i,:] = gs[i]
    #                ### checking that k1 and k2 actual form is what is expected
    #                k1p = dot(kqm.k2icartes, (ki+kqm.shift[:3]/2.)/kqm.ndiv)
    #                k2p = dot(kqm.k2icartes, (kis[i]+kqm.shift[:3]/2.)/kqm.ndiv)
    #                if sum(abs(k1-k1p))>1e-10 or sum(abs(k2-k2p))>1e-10:
    #                    print('ERROR: expectad to have k1=', k1, '=',k1p,' and k2=', k2,'=', k2p)
    #                print(' %5d %5d'%(ik+1, jk[i]+1), '   %3d %3d %3d'%tuple(gs[i]), '%5d'%(jk_irr[i]+1), ' %.5f'%(aq,))
    #                #print('dist['+str(nntot*ik+i)+']=',aq)
    #
    #Compute_and_Save_BandOverlap_Mmn(pair, pair_umklap, distances, idistance, nbe, nbs, Irredk, case, strc, latgen, ks, kqm, pw, in1, radf, fout, DMFT1=False, debug=False)

def Compute_QGT(Resort=False):
    # M[k,b]_{m,n}   = < psi_{m,k} | e^{-i b r} |psi_{n,k+b}>
    pair = load('pair.npy')
    pair_umklap = load('pair_umklap.npy')
    with open('decode.pkl', 'rb') as f:
        decode = pickle.load(f)
        (nbs,nbe) = pickle.load(f)
    
    nbands = nbe-nbs
    M_all = load('M_mmn.npy')
    
    #M_all = zeros((shape(pair)[0],shape(pair)[1],nbands,nbands),dtype=complex)
    #with open('M_mmn.npy', 'rb') as f:
    #    for ik in range(shape(pair)[0]):
    #        for idk in range(shape(pair)[1]):
    #            try:
    #                M_all[ik,idk,:,:] = load(f)
    #            except:
    #                print('ERROR reading M_mmn.npy. all data has been read at', (ik, idk), 'but require more', shape(pair))
    #                break

        
    #nbs,nbe=0,nbands
    
    (case, nkdivs, k0shift, fout) = Initialize('a',False)

    strc = w2k.Struct(case, fout)
    latgen = w2k.Latgen(strc, fout)
    latgen.Symoper(strc, fout)
    kqm = KQmesh(nkdivs, k0shift, strc, latgen, fout)

    nnkp = kqm.ndiv[0]*kqm.ndiv[1]*kqm.ndiv[2]
    print('M_mmn',file=fout)
    ind = zeros((len(kqm.k_ind),nbe-nbs),dtype=int)
    if Resort:
        for ik in range(shape(M_all)[0]):
            irk = kqm.kii_ind[ik]
            if sum(ind[irk,:])==0:
                ind[irk,:] = list(range(nbe-nbs))
            for idk in range(shape(M_all)[1]):
                jk = pair[ik,idk]
                jrk = kqm.kii_ind[jk]
                G = pair_umklap[ik,idk]
                print('ik=', ik, 'jk=', jk, 'irk=', irk, 'jrk=', jrk, 'G_umklap=', G, '<psi|psi>=', file=fout)
                PrintMC(M_all[ik,idk],file=fout)
                
                M = M_all[ik,idk,ind[irk,:],:] # M[k,b]_{m,n} = < psi_{m,k} | e^{-i b r} |psi_{n,k+b}>
                ind_new=False
                if sum(ind[jrk,:])==0: # not yet set
                    mm = abs(M)
                    for i in range(nbe-nbs):
                        ii = unravel_index(mm.argmax(), mm.shape)
                        ind[jrk,ii[0]]=ii[1]
                        mm[ii[0],:]=0
                        mm[:,ii[1]]=0
                        #print('ind=', ind[jk,:], 'mm=',file=fout)
                        #PrintM(mm,file=fout)
                    ind_new=True
                    
                Mn = M[:,ind[jrk,:]]
                print('ind['+str(irk)+']=', ind[irk,:].tolist(), 'ind['+str(jrk)+']=', ind[jrk,:], 'ind_new=', ind_new, file=fout)
                PrintMC(Mn,file=fout)
                
                M_all[ik,idk,:,:] = Mn
            
    for irk in range(len(ind)):
        print(irk, 'ind['+str(irk)+']=', ind[irk,:], file=fout)
        
    R = zeros((nbe-nbs,3,3),dtype=complex)
    for i0 in range(kqm.ndiv[0]):
        for i1 in range(kqm.ndiv[1]):
            for i2 in range(kqm.ndiv[2]):
                cases=[ ([i0,i1,i2],[i0+1,i1,i2]),  # (k,k+e_x)
                        ([i0,i1,i2],[i0,i1+1,i2]),  # (k,k+e_y)
                        ([i0,i1,i2],[i0,i1,i2+1]),  # (k,k+e_z)
                        ([i0+1,i1,i2],[i0,i1+1,i2]),# (k+e_x,k+e_y)
                        ([i0+1,i1,i2],[i0,i1,i2+1]),# (k+e_x,k+e_z)
                        ([i0,i1+1,i2],[i0+1,i1,i2]),# (k+e_y,k+e_x)
                        ([i0,i1+1,i2],[i0,i1,i2+1]),# (k+e_y,k+e_z)
                        ([i0,i1,i2+1],[i0+1,i1,i2]),# (k+e_z,k+e_x)
                        ([i0,i1,i2+1],[i0,i1+1,i2])]# (k+e_z,k+e_y)
                icase=[]
                for k1,k2 in cases:
                    k1 = array(k1,dtype=int)
                    k2 = array(k2,dtype=int)
                    gs1 = -Shift_to_1BZ(k1,kqm.ndiv)
                    gs2 = -Shift_to_1BZ(k2,kqm.ndiv)
                    k1_bz = k1-gs1*kqm.ndiv
                    k2_bz = k2-gs2*kqm.ndiv
                    ik = kqm.k2indx(k1_bz)
                    jk = kqm.k2indx(k2_bz)
                    G = gs2-gs1
                    il = decode[ik][jk]
                    dg = pair_umklap[ik,il]-G
                    if sum(abs(dg))>0:
                        print('ERROR pair_umklap and G are different p_u=', pair_umklap[ik,il], 'G=', G)
                    if (jk!=pair[ik,il]):
                        print('ERROR jk=', jk, 'but from decode we expect', pair[ik,il])
                    icase.append([ik,il])
                
                dR=zeros((nbe-nbs,3))
                for imu in range(3):  # over x,y,z in lattice coordinates
                    for i in range(0,nbe-nbs): #  over all the relevant bands, index i
                        dR[i,imu] = 1.0 - sum(abs(M_all[icase[imu][0],icase[imu][1]][:,i])**2)  # 1-\sum_m |<psi_{m,k}|psi_{i,k+e_{imu}}>|^2
                        R[i,imu,imu] += 1.0 - sum(abs(M_all[icase[imu][0],icase[imu][1]][:,i])**2)  # 1-\sum_m |<psi_{m,k}|psi_{i,k+e_{imu}}>|^2

                tr=zeros(nbe-nbs)
                dtr=zeros(nbe-nbs)
                for imu in range(3):
                    tr += R[:,imu,imu].real
                    dtr += dR[:,imu]
                print('ik=', icase[0][0],'jk=', icase[0][1],icase[1][0],icase[2][0], 'tr=', sum(dtr), sum(tr), ';', dtr.tolist(), file=fout)
                          
                for i in range(0,nbe-nbs):
                    R[i,0,1] += M_all[icase[3][0],icase[3][1]][i,i] - sum(M_all[icase[0][0],icase[0][1]][:,i].conj()*M_all[icase[1][0],icase[1][1]][:,i]) # <psi_{i,k+e_x}|psi_{i,k+e_y}-\sum_j <psi_{j,k}|psi_{i,k+e_x}>^* * <psi_{j,k}|psi_{i,k+e_y}>
                    R[i,0,2] += M_all[icase[4][0],icase[4][1]][i,i] - sum(M_all[icase[0][0],icase[0][1]][:,i].conj()*M_all[icase[2][0],icase[2][1]][:,i]) # <psi_{i,k+e_x}|psi_{i,k+e_z}-\sum_j <psi_{j,k}|psi_{i,k+e_x}>^* * <psi_{j,k}|psi_{i,k+e_z}>
                    R[i,1,0] += M_all[icase[5][0],icase[5][1]][i,i] - sum(M_all[icase[1][0],icase[1][1]][:,i].conj()*M_all[icase[0][0],icase[0][1]][:,i]) # <psi_{i,k+e_y}|psi_{i,k+e_x}-\sum_j <psi_{j,k}|psi_{i,k+e_y}>^* * <psi_{j,k}|psi_{i,k+e_x}>
                    R[i,1,2] += M_all[icase[6][0],icase[6][1]][i,i] - sum(M_all[icase[1][0],icase[1][1]][:,i].conj()*M_all[icase[2][0],icase[2][1]][:,i]) # <psi_{i,k+e_y}|psi_{i,k+e_z}-\sum_j <psi_{j,k}|psi_{i,k+e_y}>^* * <psi_{j,k}|psi_{i,k+e_z}>
                    R[i,2,0] += M_all[icase[7][0],icase[7][1]][i,i] - sum(M_all[icase[2][0],icase[2][1]][:,i].conj()*M_all[icase[0][0],icase[0][1]][:,i]) # <psi_{i,k+e_z}|psi_{i,k+e_x}-\sum_j <psi_{j,k}|psi_{i,k+e_z}>^* * <psi_{j,k}|psi_{i,k+e_x}>
                    R[i,2,1] += M_all[icase[8][0],icase[8][1]][i,i] - sum(M_all[icase[2][0],icase[2][1]][:,i].conj()*M_all[icase[1][0],icase[1][1]][:,i]) # <psi_{i,k+e_z}|psi_{i,k+e_y}-\sum_j <psi_{j,k}|psi_{i,k+e_z}>^* * <psi_{j,k}|psi_{i,k+e_y}>

    del M_all
    
    R *= 1/nnkp
    for i in range(3):
        for j in range(3):
            # because 1/(dk_i*dk_j) = N_i * N_j
            R[:,i,j] *= kqm.ndiv[i]*kqm.ndiv[j]

    dsm=0
    for i in range(nbe-nbs):
        dsm += sum([R[i,imu,imu] for imu in range(3) ])
    print('Tr(R)=', dsm, file=fout)
    for i in range(nbe-nbs):
        print('R for band i=', i, sum([R[i,imu,imu] for imu in range(3) ]), file=fout)
        PrintMC(R[i], file=fout)

    
    k2cartes_m1 = diag(1/diag(kqm.k2cartes))
    prefactor = k2cartes_m1**2 * (8*pi**2/latgen.Vol)
    gij =  linalg.inv(dot(latgen.br2.T,latgen.br2)) @ kqm.k2cartes**2
    renorm = (gij[0,0]+gij[1,1]+gij[2,2])/3.
    gij /= renorm
    prefactor *= renorm
    
    #print('prefactor=', prefactor,file=fout)
    #print('eta^{ij}=', gij,file=fout)
    #print('prefactor*eta^{ij}=', gij @ prefactor,file=fout)
    #print('expect=', linalg.inv(dot(latgen.br2.T,latgen.br2)) * (8*pi**2/latgen.Vol), file=fout)
    
    Omega = zeros(shape(R))
    G = zeros(shape(R))
    for i in range(nbe-nbs):
        Omega[i,:] = ((R[i,:,:]-R[i,:,:].T)*1j).real
        G[i,:,:]     = prefactor @ (0.5*(R[i,:,:]+R[i,:,:].T).real) # *(8*pi**2)/latgen.Vol
    

    for i in range(nbe-nbs):
        print('Omega_{lattice} for band i=', i,file=fout)
        PrintM(Omega[i],file=fout)
        
    print('eta^{ij}=',file=fout)
    PrintM(gij,file=fout)
    
    dsm=0
    dtr=0
    for i in range(nbe-nbs):
        dsm += sum([G[i,imu,imu] for imu in range(3)])
        dtr += sum(G[i,:,:]*gij[:,:].T)
    dsm /= nbe-nbs
    dtr /= nbe-nbs
    for i in range(nbe-nbs):
        print('G_{lattice} for band i=', i, file=fout)
        PrintM(G[i],file=fout)
    print('bands considered '+str(nbs)+'<= i < '+str(nbe), file=fout)
    print('Tr(G_{lattice})/N_bands=', dsm, file=fout)
    print('Tr(G_{cartesian})/N_bands=', dtr, file=fout)

    
    
if __name__ == '__main__':
    Compute_mmn(nbs=0,nbe=0,cmpEF=True)
    #Compute_mmn(nbs=0,nbe=4,cmpEF=True)
    if mrank==master:
        Compute_QGT(Resort=False)
    
