import sys, os, re
from datetime import date
from numpy import *
import glob
import pickle

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
    nbs_nbe = array(loadtxt('nbs_nbe.dat'),dtype=int)
    xk = Ene[0,:]
    Ene = Ene[1:,:]
    maxv = None
    with open('pairs.pkl', 'rb') as f:
        pairs = pickle.load(f)
        [nbs0,nbe0,tnbs,tnbe,latgen_Vol] = pickle.load(f)
    
    Logarithmic = [True,False,False]
    percent_removed=[0,0,0]
    Qcolor = [True for i in range(len(nbs_nbe))]
    #Qcolor = [False,True,False]
    #ymin,ymax = -12,12
    ymin,ymax = -12,7.4
    #ymin,ymax=-3,3
    #ymin,ymax=-2.,2.6
    #ymin,ymax=-2.5,2.5
    cmap = mpl.cm.coolwarm

    if os.path.isfile('ppar.dat'):
        exec(open("ppar.dat").read())
        print('Logarithmic=', Logarithmic)
        print('maxv=', maxv)
        #print('Qcolor=', Qcolor)
        print('ymin,ymax=', ymin,ymax)
    
    QColor=[]
    for i in range(len(nbs_nbe)):
        if Qcolor[i]:
            QColor += [True for j in range(nbs_nbe[i,0],nbs_nbe[i,1])]
        else:
            QColor += [False for j in range(nbs_nbe[i,0],nbs_nbe[i,1])]
            
    case = Get_case()
    (kpnt,name_kp) = Read_klist(case+'.klist_band')
    
    print('Qcolor=', Qcolor)
    print('shape(Ene)=', shape(Ene))
    
    ne = 100
    
    Nplots=3
    Ratio=False
    
    _qgt_ = loadtxt('QGT.dat').T    # QGT is just quantum metric
    qgt = [_qgt_[2:,:]]
    if not (os.path.isfile('GeD.dat') and os.path.isfile('CoD.dat')):
        Nplots=1
        Ratio=False
        
    if Nplots>=2:
        _qpm_ = loadtxt('GeD.dat').T    # GeD is geometric stifness
        geD = _qpm_[2:,:]
        qgt+= [geD]
    if Nplots>=3:
        _cod_ = loadtxt('CoD.dat').T   # CoD is conventional sifness
        coD = _cod_[2:,:]
        qgt += [coD]
    
    ne = min(min(ne,len(Ene)), int(round((len(qgt[0]))/3)))
    
    print('shape(gqt)=', shape(qgt[0]), len(qgt))
    print('nbs_nbe=', nbs_nbe)
    print('nbs0=', nbs0, 'nbe0=', nbe0)
    lbl = ['QMT', 'D_geom', 'D_conv']
    
    fig,ax = subplots(nrows=4,ncols=Nplots,figsize=(7*Nplots,10), # (15,10)
                        gridspec_kw={'width_ratios': [1/Nplots]*Nplots, 'height_ratios': [10,10,10,1], 'wspace' : 0.3, 'hspace' : 0.2})
    
    if Nplots==1:
        ax=[[ax[0]],[ax[1]],[ax[2]],[ax[3]]]

    for ipl in range(Nplots):
        miv,mav = 1e10,0
        nb_start = nbs_nbe[0,0]
        print('nb_start=', nb_start)

        n1 = nbs_nbe[-1,1]-nbs_nbe[0,0]
        miv = min(qgt[ipl][:3*n1,:].ravel())
        mav = max(qgt[ipl][:3*n1,:].ravel())
        ht, bin_edges = histogram(qgt[ipl][:3*n1,:].ravel(),bins=5000)
        xh = 0.5*(bin_edges[1:]+bin_edges[:-1])
        cums = cumsum(ht)/sum(ht)
        iis = searchsorted(cums, percent_removed[ipl])
        iie = searchsorted(cums, 1-percent_removed[ipl])
        print(ipl, 'with percent_removed=', percent_removed[ipl], 'we determine cutoff at =', str(xh[iis])+':'+str(xh[iie]), 'instead of', str(miv)+':'+str(mav))
        miv = xh[iis]
        mav = xh[iie]
        
        if maxv is not None:
            mav=maxv
        
        #for i in range(len(nbs_nbe)):
            #if Qcolor[i]:
            #    n0, n1 = nbs_nbe[i,0]-nb_start, nbs_nbe[i,1]-nb_start
            #    m = min(qgt[ipl][3*n0:3*n1,:].ravel())
            #    miv = min(miv,m)
            #    m = max(qgt[ipl][3*n0:3*n1,:].ravel())
            #    mav = max(mav,m)
            #
            #    ht, bin_edges = histogram(qgt[ipl][3*n0:3*n1,:].ravel(),bins=5000)
            #    xh = 0.5*(bin_edges[1:]+bin_edges[:-1])
            #    cums = cumsum(ht)/sum(ht)
            #    iic = searchsorted(cums, 1-percent_removed[ipl])
            #    print(nbs_nbe[i], 'with percent_removed=', percent_removed[ipl], 'we determine cutoff at =', xh[iic], 'instead of max', mav)
            #    mav = xh[iic]
                
        small=1e-6
        print('min=', miv,'max=', mav, 'ne=', ne)
        #mav = mav*(1.-percent_removed[ipl])
        #print('changed to : min=', miv, 'max=', mav, 'ne=', ne)
        
        for i in range(0,ne):
            if (QColor[i]):
                if Logarithmic[ipl]:
                    rx = log(abs(qgt[ipl][3*i+0,:])+small)/log(mav+small)
                    ry = log(abs(qgt[ipl][3*i+1,:])+small)/log(mav+small)
                    rz = log(abs(qgt[ipl][3*i+2,:])+small)/log(mav+small)
                else:
                    rx = qgt[ipl][3*i+0,:]/mav
                    ry = qgt[ipl][3*i+1,:]/mav
                    rz = qgt[ipl][3*i+2,:]/mav
                colx=cmap([0.5*(rx[ik]+rx[ik+1]) for ik in range(len(xk)-1)])
                coly=cmap([0.5*(ry[ik]+ry[ik+1]) for ik in range(len(xk)-1)])
                colz=cmap([0.5*(rz[ik]+rz[ik+1]) for ik in range(len(xk)-1)])
                for ik in range(len(xk)-1):
                    ax[0][ipl].plot([xk[ik],xk[ik+1]],[Ene[i,ik],Ene[i,ik+1]],color=colx[ik])
                    ax[1][ipl].plot([xk[ik],xk[ik+1]],[Ene[i,ik],Ene[i,ik+1]],color=coly[ik])
                    ax[2][ipl].plot([xk[ik],xk[ik+1]],[Ene[i,ik],Ene[i,ik+1]],color=colz[ik])
            else:
                ax[0][ipl].plot(xk,Ene[i],'k-')
                ax[1][ipl].plot(xk,Ene[i],'k-')
                ax[2][ipl].plot(xk,Ene[i],'k-')
                
        
        dire=['$M_{XX}$','$M_{YY}$','$M_{ZZ}$']
        for i in range(3):
            tax = ax[i][ipl]
            tax.set_ylim([ymin,ymax])
            tax.set_xlim([xk[0],xk[-1]])
            tax.set_ylabel(dire[i])
        
        if ymin is not None:
            _col_ = 'k'
            for i in range(3):
                tax = ax[i][ipl]
                for wi,name in name_kp.items():
                    cs=tax.plot([xk[wi],xk[wi]], [ymin,ymax], _col_+'-')
                tax.plot([xk[0],xk[-1]],[0,0], _col_+':')
                tax.set_ylim([ymin,ymax])
                
                tax.set_xticks( [xk[wi] for wi in name_kp], [name_kp[wi] for wi in name_kp], fontsize='x-large' )
        
        if Logarithmic[ipl]:
            _miv_ = miv
            if miv<0: _miv_=0
            norm = mpl.colors.LogNorm(vmin=_miv_+small, vmax=mav+small)
        else:
            norm = mpl.colors.Normalize(vmin=miv, vmax=mav)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=ax[3][ipl], orientation='horizontal', label=lbl[ipl])    
    
    show()


    if Ratio:
        Percent=zeros(((nbs_nbe[-1][1]-nbs_nbe[0][0]),shape(qgt[1])[1]))
        nb_start = nbs_nbe[0,0]
        miv,mav = 1e10,0
        for i in range(len(nbs_nbe)):
            n0, n1 = nbs_nbe[i,0]-nb_start, nbs_nbe[i,1]-nb_start
            for l in range(n0,n1):
                Dgeom=(qgt[1][3*l,:]+qgt[1][3*l+1,:]+qgt[1][3*l+2,:])
                Dconv=(qgt[2][3*l,:]+qgt[2][3*l+1,:]+qgt[2][3*l+2,:])
                Pc = Dgeom/(Dgeom+Dconv)
                m = min(Pc.ravel())
                miv = min(miv,m)
                m = max(Pc.ravel())
                mav = max(mav,m)
                Percent[l,:] = Pc
        print('min=', miv, 'max=', mav)
                
        fig,ax = subplots(nrows=2,ncols=1,gridspec_kw={'height_ratios': [10,1]})

        #for i in range(len(nbs_nbe)):
        #    if Qcolor[i]:
        #        n0, n1 = nbs_nbe[i,0]-nb_start, nbs_nbe[i,1]-nb_start
        #        for l in range(n0,n1):
        #            t = (qgt[ipl][3*l,:]+qgt[ipl][3*l+1,:]+qgt[ipl][3*l+2,:])
        #            m = min(t.ravel())
        #            miv = min(miv,m)
        #            m = max(t.ravel())
        #            mav = max(mav,m)
        
        for i in range(0,ne):
            if (QColor[i]):
                rt = Percent[i,:]/mav
                colt=cmap([0.5*(rt[ik]+rt[ik+1]) for ik in range(len(xk)-1)])
                for ik in range(len(xk)-1):
                    ax[0].plot([xk[ik],xk[ik+1]],[Ene[i,ik],Ene[i,ik+1]],color=colt[ik])
            else:
                ax[0].plot(xk,Ene[i],'k-')

        tax = ax[0]
        tax.set_ylim([ymin,ymax])
        tax.set_xlim([xk[0],xk[-1]])
        tax.set_ylabel('D_geom/(D_geom+D_conv)')
        
        if ymin is not None:
            _col_ = 'k'
            for wi,name in name_kp.items():
                cs=tax.plot([xk[wi],xk[wi]], [ymin,ymax], _col_+'-')
            tax.plot([xk[0],xk[-1]],[0,0], _col_+':')
            tax.set_ylim([ymin,ymax])
            tax.set_xticks( [xk[wi] for wi in name_kp], [name_kp[wi] for wi in name_kp], fontsize='x-large' )
        
        norm = mpl.colors.Normalize(vmin=miv, vmax=mav)
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=ax[1], orientation='horizontal', label='percent')
        
        show()

