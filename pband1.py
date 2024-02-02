import sys, os, re
from datetime import date
from numpy import *
import glob

from cmn import Ry2H,H2eV,Ry2eV,H2Ry,Br2Ang
from pylab import *
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib import gridspec

mingle_names={'W':'$W$','L':'$L$','LAMBDA':'$\Lambda$','GAMMA':'$\Gamma$','DELTA':'$\Delta$','X':'$X$','Z':'$Z$','W':'$W$','K':'$K$'}
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
            if legnd in mingle_names:
                legnd = mingle_names[legnd]
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

if __name__ == '__main__':
    Ene = loadtxt('eQGT.dat').T
    _qgt_ = loadtxt('QGT.dat').T
    nbs_nbe = array(loadtxt('nbs_nbe.dat'),dtype=int)
    qgt = _qgt_[2:,:]
    xk = Ene[0,:]
    Ene = Ene[1:,:]
    
    Logarithmic = False #True
    #Qcolor = [True for i in range(len(nbs_nbe))]
    Qcolor = [True,False]
    ymin,ymax = -12,12
    #ymin,ymax = -6,6
    #ymin,ymax=-3,3
    #ymin,ymax=-2.,2.6
    #ymin,ymax=-2.5,2.5
    cmap = mpl.cm.jet


    ne = 100
    
    ne = min(min(ne,len(Ene)), int(round((len(qgt))/3)))
    
    print('shape(Ene)=', shape(Ene))
    print('shape(gqt)=', shape(qgt))
    print(Qcolor)

    miv,mav = 1e10,0
    for i in range(len(nbs_nbe)):
        if Qcolor[i]:
            m = min(qgt[3*nbs_nbe[i,0]:3*nbs_nbe[i,1],:].ravel())
            miv = min(miv,m)
            m = max(qgt[3*nbs_nbe[i,0]:3*nbs_nbe[i,1],:].ravel())
            mav = max(mav,m)
            
    #miv = min(qgt[:3*ne,:].ravel())
    #mav = max(qgt[:3*ne,:].ravel())
    
    #mav=5
    print('min=', miv,'max=', mav, 'ne=', ne)

    QColor=[]
    for i in range(len(nbs_nbe)):
        if Qcolor[i]:
            QColor += [True for j in range(nbs_nbe[i,0],nbs_nbe[i,1])]
        else:
            QColor += [False for j in range(nbs_nbe[i,0],nbs_nbe[i,1])]
            
    fig,ax = subplots(nrows=4,ncols=1,figsize=(15,10),
                        gridspec_kw={'width_ratios': [1], 'height_ratios': [10,10,10,1], 'wspace' : 0.3, 'hspace' : 0.2})
    for i in range(0,ne):
        if (QColor[i]):
            if Logarithmic:
                rx = log(1+qgt[3*i+0,:])/log(1+mav)
                ry = log(1+qgt[3*i+1,:])/log(1+mav)
                rz = log(1+qgt[3*i+2,:])/log(1+mav)
            else:
                rx = qgt[3*i+0,:]/mav
                ry = qgt[3*i+1,:]/mav
                rz = qgt[3*i+2,:]/mav
            colx=cmap([0.5*(rx[ik]+rx[ik+1]) for ik in range(len(xk)-1)])
            coly=cmap([0.5*(ry[ik]+ry[ik+1]) for ik in range(len(xk)-1)])
            colz=cmap([0.5*(rz[ik]+rz[ik+1]) for ik in range(len(xk)-1)])
            for ik in range(len(xk)-1):
                ax[0].plot([xk[ik],xk[ik+1]],[Ene[i,ik],Ene[i,ik+1]],color=colx[ik])
                ax[1].plot([xk[ik],xk[ik+1]],[Ene[i,ik],Ene[i,ik+1]],color=coly[ik])
                ax[2].plot([xk[ik],xk[ik+1]],[Ene[i,ik],Ene[i,ik+1]],color=colz[ik])
        else:
            ax[0].plot(xk,Ene[i],'k-')
            ax[1].plot(xk,Ene[i],'k-')
            ax[2].plot(xk,Ene[i],'k-')
            
    
    dire=['$M_{XX}$','$M_{YY}$','$M_{ZZ}$']
    for i in range(3):
        tax = ax[i]
        tax.set_ylim([ymin,ymax])
        tax.set_xlim([xk[0],xk[-1]])
        tax.set_ylabel(dire[i])

    case = Get_case()
    (kpnt,name_kp) = Read_klist(case+'.klist_band')
    if ymin is not None:
        _col_ = 'k'
        for i in range(3):
            tax = ax[i]
            for wi,name in name_kp.items():
                cs=tax.plot([xk[wi],xk[wi]], [ymin,ymax], _col_+'-')
            tax.plot([xk[0],xk[-1]],[0,0], _col_+':')
            tax.set_ylim([ymin,ymax])
            
            tax.set_xticks( [xk[wi] for wi in name_kp], [name_kp[wi] for wi in name_kp], fontsize='x-large' )

    if Logarithmic:
        norm = mpl.colors.LogNorm(vmin=1+miv, vmax=1+mav)
    else:
        norm = mpl.colors.Normalize(vmin=miv, vmax=mav)
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=ax[3], orientation='horizontal', label='QMT')    
    
    show()
