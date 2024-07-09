import math
import array
import numpy as np
import ROOT

def GetLogBinning(nbins,xmin,xmax):
    logmin  = math.log10(xmin)
    logmax  = math.log10(xmax)
    logbinwidth = (logmax-logmin)/nbins
    # Bin edges
    xbins = [xmin,] #the lowest edge first
    for i in range(1,nbins+1):
        xbins.append( ROOT.TMath.Power( 10,(logmin + i*logbinwidth) ) )
    arrxbins = array.array("d", xbins)
    return nbins, arrxbins

############################
### for normal histos
Emin = 3e-1 #TODO: SHOULD BE 2!!!
Emax = 100
n_E  = 50

dEmin = 1e-7
dEmax = 1e0
n_dE  = 500

dxmin = 1e-7
dxmax = 2e2
n_dx  = 500

dxinvmin = 1./dxmax
dxinvmax = 1./dxmin
n_dxinv  = 500

dRmin = 1e-10
dRmax = 1e-1
n_dR  = 500

dLmin = 1e-7
dLmax = 2e2
n_dL  = 500

dRinvmin = 1./dRmax
dRinvmax = 1./dRmin
n_dRinv  = 500

dEdxmin = 3e-6
dEdxmax = 1e5
n_dEdx  = 500

nEbins,Ebins         = GetLogBinning(n_E, Emin,Emax)
ndEbins,dEbins       = GetLogBinning(n_dE,dEmin,dEmax)
ndxbins,dxbins       = GetLogBinning(n_dx,dxmin,dxmax)
ndLbins,dLbins       = GetLogBinning(n_dL,dLmin,dLmax)
ndxinvbins,dxinvbins = GetLogBinning(n_dxinv,dxinvmin,dxinvmax)
ndRbins,dRbins       = GetLogBinning(n_dR,dRmin,dRmax)
ndRinvbins,dRinvbins = GetLogBinning(n_dRinv,dRinvmin,dRinvmax)
ndEdxbins,dEdxbins   = GetLogBinning(n_dEdx,dEdxmin,dEdxmax)

####################
### for slicees
n_small_dE    = 200
n_small_E     = 50
n_small_dx    = 50
n_small_dL    = 50
n_small_dxinv = 50

nEbins_small,Ebins_small         = GetLogBinning(n_small_E,Emin,Emax)
ndxbins_small,dxbins_small       = GetLogBinning(n_small_dx,dxmin,dxmax)
ndxinvbins_small,dxinvbins_small = GetLogBinning(n_small_dxinv,dxinvmin,dxinvmax)
ndLbins_small,dLbins_small       = GetLogBinning(n_small_dL,dLmin,dLmax)
ndEbins_small,dEbins_small       = GetLogBinning(n_small_dE,dEmin,dEmax)

####################
### others
nEbins_forDp,Ebins_forDp = GetLogBinning(500,0.1,100)
ndxbins_forDp,dxbins_forDp = GetLogBinning(500,1e-7,1.5e2)
   
