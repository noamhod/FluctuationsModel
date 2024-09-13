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

def split_bins(bin_edges, split_indices):
    ### bin_edges     = [0,1,2,3,4,5] <-- example input
    ### N_edges       = 6
    ### N_bins        = 5 (=N_edges-1)
    ### bins          = [(0,1), (1,2), (2,3), (3,4), (4,5)]
    ### split_indices = [2,5] <-- example input
    ### new_bins      = [(0,1), (1,1.5), (1.5,2), (2,3), (3,4), (4,4.5), (4.5,5)]
    ### new_bin_edges = [0,1,1.5,2,3,4,4.5,5]
    ### new_N_edges   = 8
    ### new_N_bins    = 7 (=new_N_edges-1)

    N_edges = len(bin_edges)
    N_bins  = N_edges-1

    ### safety checks
    split_indices.sort()
    imin = split_indices[0]
    imax = split_indices[-1]
    if(imin<1):
        print(f"Error: imin={imin}. Quitting")
        quit()
    if(imax>N_bins):
        print(f"Error: imax={imax}. Quitting")
        quit()

    ### now do the edge appending
    new_bin_edges = []
    for iedge in range(N_edges):
        ibin = iedge+1
        binL = bin_edges[iedge]
        new_bin_edges.append(binL)
        if(ibin in split_indices):
            binR = bin_edges[iedge+1]
            new_edge = (binL+binR)/2.
            new_bin_edges.append(new_edge)
    arrxbins = array.array("d", new_bin_edges)
    return arrxbins
    # return new_bin_edges


############################
### for normal histos
Emin = 0.46 #3e-1
Emax = 100
n_E  = 50
n_E_big = 500

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
nEbins_big,Ebins_big = GetLogBinning(n_E_big, Emin,Emax)
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
n_small_dL    = 50
n_small_dx    = 50
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
   
