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

from cmn import Ry2H,H2eV,Ry2eV,H2Ry,Br2Ang
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
import matplotlib.cm as cm
#from matplotlib.pyplot import colorbar
import matplotlib as mpl
from matplotlib import gridspec

mingle_names={'W':'$W$','L':'$L$','LAMBDA':'$\Lambda$','GAMMA':'$\Gamma$','DELTA':'$\Delta$','X':'$X$','Z':'$Z$','W':'$W$','K':'$K$'}
def Find_EF(case):
    scf = open(case+'.scf','r').readlines()
    for line in scf[::-1]:
        if line[:4]==':FER':
            EF = float(line.split()[-1])
            return EF
    return 0

def Read_klist(case):
    wg = glob.glob(case+'.klist_band')
    if len(wg)>0:
        fg = open(wg[0], 'r')
        wkpointi=[]
        wkpoints=[]
        for il,line in enumerate(fg):
            if line[:3]=='END': break
            com = line[:10].split()
            if com:
                legnd=line.split()[0]
                if legnd in mingle_names:
                    legnd = mingle_names[legnd]
                wkpoints.append(legnd)
                wkpointi.append(il)
        #print(wkpointi)
        #print(wkpoints)
    return (wkpointi,wkpoints)
if __name__ == '__main__':
    qmd = loadtxt('all_qgm.dat')
    
    nbs,nbe = 50,53
    #ymin,ymax=None,None
    ymin,ymax = -12,6
    
    fout = open('pband.out', 'w')
    struct_names = glob.glob('*.struct')
    if len(struct_names)==0:
        print('ERROR : Could not find a *.struct file present')
        sys.exit(0)
    else:
        case = struct_names[0].split('.')[0]

    EF = Find_EF(case)
        
    strc = w2k.Struct(case, fout)
    (klist2, wegh, ebnd2, hsrws, knames) = w2k.Read_energy_file(case+'.energy_band', strc, fout, give_kname=True)

    xk = zeros(len(klist2))
    for ik in range(1,len(klist2)):
        xk[ik] = xk[ik-1] + linalg.norm(array(klist2[ik])-array(klist2[ik-1]))

    nbm = min([len(eke) for eke in ebnd2])
    Ene = zeros((len(ebnd2),nbm))
    for ik in range(len(ebnd2)):
        Ene[ik,:] = ebnd2[ik][:nbm]
    Ene -= EF
    Ene *= Ry2eV

    col = ones(nbm)
    for i,c in qmd:
        col[int(i)]=c
    vmin = min(col)
    vmax = max(col)
    cl = (col-vmin)/(vmax-vmin)
    cmap = mpl.cm.jet
    col = cmap(cl)
    
    print('col=', col)
    fig, (ax,ax2) = subplots(nrows=2,ncols=1,figsize=(8,6), gridspec_kw={'width_ratios': [1], 'height_ratios': [15,1], 'wspace' : 0.3, 'hspace' : 0.15})
    ax.set_title('bands grouped (0,9)')
    for ib in range(nbm):
        lw=2
        #if (ib>=nbs and ib<nbe): lw=3
        ax.plot(xk, Ene[:,ib], lw=lw, color=col[ib])

    (wkpointi,wkpoints) = Read_klist(case)

    if ymin is not None:
        _col_ = 'k'
        for wi in wkpointi:
            cs=ax.plot([xk[wi],xk[wi]], [ymin,ymax], _col_+'-')
        
        ax.plot([xk[0],xk[-1]],[0,0], _col_+':')
        ax.set_ylim([ymin,ymax])
    
    ax.set_xlim([xk[0],xk[-1]])
    ax.set_xticks( [xk[wi] for wi in wkpointi], wkpoints, fontsize='x-large' )

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=ax2, orientation='horizontal', label='QMT')    

    show()
