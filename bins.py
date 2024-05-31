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

Emin = 3e-1 #TODO: SHOULD BE 2!!!
Emax = 100
dEmin = 1e-7
dEmax = 1e0
dxmin = 1e-7
dxmax = 1e3
dxinvmin = 1./dxmax
dxinvmax = 1./dxmin
dRmin = 1e-15
dRmax = 1e-1
dRinvmin = 1./dRmax
dRinvmax = 1./dRmin
dEdxmin = 3e-6
dEdxmax = 1e5

nEbins,Ebins         = GetLogBinning(50,Emin,Emax)
ndxbins,dxbins       = GetLogBinning(500,dxmin,dxmax)
ndxinvbins,dxinvbins = GetLogBinning(500,dxinvmin,dxinvmax)
ndEbins,dEbins       = GetLogBinning(500,dEmin,dEmax)
ndRbins,dRbins       = GetLogBinning(500,dRmin,dRmax)
ndRinvbins,dRinvbins = GetLogBinning(500,dRinvmin,dRinvmax)
ndEdxbins,dEdxbins   = GetLogBinning(500,dEdxmin,dEdxmax)

n_small_dE    = 200
n_small_E     = 120
n_small_dx    = 120
n_small_dxinv = 120
nEbins_small,Ebins_small         = GetLogBinning(n_small_E,Emin,Emax)
ndxbins_small,dxbins_small       = GetLogBinning(n_small_dx,dxmin,dxmax)
ndxinvbins_small,dxinvbins_small = GetLogBinning(n_small_dxinv,dxinvmin,dxinvmax)
ndEbins_small,dEbins_small       = GetLogBinning(n_small_dE,dEmin,dEmax)

nEbins_forDp,Ebins_forDp = GetLogBinning(500,0.1,100)
ndxbins_forDp,dxbins_forDp = GetLogBinning(500,1e-7,1.5e2)
   
