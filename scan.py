import time
import pickle
import math
import array
import numpy as np
import pandas as pd
import ROOT
import units as U
import constants as C
import material as mat
import particle as prt
import bins
import fluctuations as flct
import hist
import model
import multiprocessing as mp
import pickle

import tracemalloc

ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.15)


### model shapes defined later as global
parallelize = True

################################
### the of all pickle files
pklpath = "/Users/noamtalhod/tmp/pkl"


##############################################################
##############################################################
##############################################################
### functions for the submission of model calculation

def get_continuous_slice(ie,il,hgrid):
    label_E     = str(ie)
    label_dL    = str(il)
    label       = "E"+label_E+"_dL"+label_dL
    E           = hgrid.GetXaxis().GetBinCenter(ie)*U.MeV2eV # eV
    L           = hgrid.GetYaxis().GetBinCenter(il)*U.um2cm  # cm
    return label, E, L

def get_secondaries_slice(ie,hgrid):
    label_E = str(ie)
    label   = "E"+label_E
    E       = hgrid.GetXaxis().GetBinCenter(ie)*U.MeV2eV # eV
    return label, E

def get_edges_continuous(ie,il,hgrid):
    Emin        = hgrid.GetXaxis().GetBinLowEdge(ie)
    Emax        = hgrid.GetXaxis().GetBinUpEdge(ie)
    Lmin        = hgrid.GetYaxis().GetBinLowEdge(il)
    Lmax        = hgrid.GetYaxis().GetBinUpEdge(il)
    slice_edges = {"E":[Emin,Emax], "L":[Lmin,Lmax]}
    return slice_edges

def get_edges_secondaries(ie,hgrid):
    Emin        = hgrid.GetXaxis().GetBinLowEdge(ie)
    Emax        = hgrid.GetXaxis().GetBinUpEdge(ie)
    slice_edges = {"E":[Emin,Emax]}
    return slice_edges


# def get_slice_shapes(E,L,P,label,slice_edges):
#     if(parallelize):
#         lock = mp.Lock()
#         lock.acquire()
#     start1 = time.time()
#     Mod = model.Model(L,E,P)
#     Mod.set_fft_sampling_pars_rotem(N_t_bins=10000000,frac=0.01)
#     # Mod.set_all_shapes()
#     Mod.set_continuous_shapes()
#     Mod.set_secondaries_shapes()
#     end1 = time.time()
#     elapsed1 = end1-start1
#
#     start2 = time.time()
#     ############################
#     ### output files
#     pklname = f"{pklpath}/slice_{label}.pkl"
#     fpkl = open(pklname,"wb")
#     ############################
#
#     ### write the sapes to the pickle file
#     data = { "Label":label,
#              "Slice_E_min":slice_edges["E"][0],
#              "Slice_E_max":slice_edges["E"][1],
#              "Slice_dL_min":slice_edges["L"][0],
#              "Slice_dL_max":slice_edges["L"][1],
#              "Pars":P,
#              "cnt_cdf_arrx":Mod.cnt_cdfs_scaled_arrx,
#              "cnt_cdf_arrsy":Mod.cnt_cdfs_scaled_arrsy,
#              "sec_cdf_arrx":Mod.sec_cdfs_arrx,
#              "sec_cdf_arrsy":Mod.sec_cdfs_arrsy }
#     pickle.dump(data, fpkl, protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
#     fpkl.close()
#     end2 = time.time()
#     elapsed2 = end2-start2
#
#     ### clean up the model class
#     ### for reasonable memory usage
#     del Mod
#
#     print(f"Finished slice: {label} at (E,dL)=({E*U.eV2MeV:.3f} MeV,{L*U.cm2um:.6f} um), model obtained within {elapsed1:.2f} [s] and saved in {elapsed2:.2f} [s]")
#     if(parallelize): lock.release()

def get_slice_continuous_shapes(E,L,P,label,slice_edges):
    if(parallelize):
        lock = mp.Lock()
        lock.acquire()
    start1 = time.time()
    Mod = model.Model(L,E,P)
    Mod.set_fft_sampling_pars_rotem(N_t_bins=10000000,frac=0.01)
    Mod.set_continuous_shapes()
    end1 = time.time()
    elapsed1 = end1-start1
    
    start2 = time.time()
    ############################
    ### output files
    pklname = f"{pklpath}/slice_cnt_{label}.pkl"
    fpkl = open(pklname,"wb")
    ############################

    ### write the sapes to the pickle file
    data = { "Label":label,
             "Slice_E_min":slice_edges["E"][0],
             "Slice_E_max":slice_edges["E"][1],
             "Slice_dL_min":slice_edges["L"][0],
             "Slice_dL_max":slice_edges["L"][1],
             "Pars":P,
             "cnt_cdf_arrx":Mod.cnt_cdfs_scaled_arrx,
             "cnt_cdf_arrsy":Mod.cnt_cdfs_scaled_arrsy}
    pickle.dump(data, fpkl, protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
    fpkl.close()
    end2 = time.time()
    elapsed2 = end2-start2
    
    ### clean up the model class 
    ### for reasonable memory usage
    del Mod
    
    print(f"Finished continuous slice: {label} at (E,dL)=({E*U.eV2MeV:.3f} MeV,{L*U.cm2um:.6f} um), model obtained within {elapsed1:.2f} [s] and saved in {elapsed2:.2f} [s]")
    if(parallelize): lock.release()

def get_slice_secondaries_shapes(E,LMID,P,label,slice_edges):
    if(parallelize): 
        lock = mp.Lock()
        lock.acquire()
    start1 = time.time()
    Mod = model.Model(LMID,E,P)
    # Mod.set_fft_sampling_pars_rotem(N_t_bins=10000000,frac=0.01)
    Mod.set_secondaries_shapes()
    end1 = time.time()
    elapsed1 = end1-start1
    
    start2 = time.time()
    ############################
    ### output files
    pklname = f"{pklpath}/slice_sec_{label}.pkl"
    fpkl = open(pklname,"wb")
    ############################

    ### write the sapes to the pickle file
    data = { "Label":label,
             "Slice_E_min":slice_edges["E"][0],
             "Slice_E_max":slice_edges["E"][1],
             "Pars":P,
             "sec_cdf_arrx":Mod.sec_cdfs_arrx,
             "sec_cdf_arrsy":Mod.sec_cdfs_arrsy }
    pickle.dump(data, fpkl, protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
    fpkl.close()
    end2 = time.time()
    elapsed2 = end2-start2
    
    ### clean up the model class 
    ### for reasonable memory usage
    del Mod
    
    print(f"Finished secondaries slice: {label} at E=({E*U.eV2MeV:.3f} MeV, model obtained within {elapsed1:.2f} [s] and saved in {elapsed2:.2f} [s]")
    if(parallelize): lock.release()

def collect_errors(error):
    ### https://superfastpython.com/multiprocessing-pool-error-callback-functions-in-python/
    print(f'Error: {error}', flush=True)

def save_continuous_grid(h2D):
    arrE_mid = np.zeros( h2D.GetNbinsX() )
    arrE_min = np.zeros( h2D.GetNbinsX() )
    arrE_max = np.zeros( h2D.GetNbinsX() )
    xaxis = h2D.GetXaxis()
    yaxis = h2D.GetYaxis()
    for ie in range(1,h2D.GetNbinsX()+1):
        EE   = xaxis.GetBinCenter(ie)
        Emin = xaxis.GetBinLowEdge(ie)
        Emax = xaxis.GetBinUpEdge(ie)
        arrE_mid[ie-1] = EE
        arrE_min[ie-1] = Emin
        arrE_max[ie-1] = Emax
    arrdL_mid = np.zeros( h2D.GetNbinsY() )
    arrdL_min = np.zeros( h2D.GetNbinsY() )
    arrdL_max = np.zeros( h2D.GetNbinsY() )
    for il in range(1,h2D.GetNbinsY()+1):
        LL    = yaxis.GetBinCenter(il)
        dLmin = yaxis.GetBinLowEdge(il)
        dLmax = yaxis.GetBinUpEdge(il)
        arrdL_mid[il-1] = LL
        arrdL_min[il-1] = dLmin
        arrdL_max[il-1] = dLmax
    slice_arrays = {"arrE":arrE_mid,"arrE_min":arrE_min,"arrE_max":arrE_max, "arrdL":arrdL_mid,"arrdL_min":arrdL_min,"arrdL_max":arrdL_max}
    fpkl = open("scan_continuous_grid.pkl","wb")
    pickle.dump(slice_arrays, fpkl, protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
    fpkl.close()

def save_secondaries_grid(h1D):
    arrE_mid = np.zeros( h1D.GetNbinsX() )
    arrE_min = np.zeros( h1D.GetNbinsX() )
    arrE_max = np.zeros( h1D.GetNbinsX() )
    xaxis = h1D.GetXaxis()
    for ie in range(1,h1D.GetNbinsX()+1):
        EE   = xaxis.GetBinCenter(ie)
        Emin = xaxis.GetBinLowEdge(ie)
        Emax = xaxis.GetBinUpEdge(ie)
        arrE_mid[ie-1] = EE
        arrE_min[ie-1] = Emin
        arrE_max[ie-1] = Emax
    slice_arrays = {"arrE":arrE_mid,"arrE_min":arrE_min,"arrE_max":arrE_max}
    fpkl = open("scan_secondaries_grid.pkl","wb")
    pickle.dump(slice_arrays, fpkl, protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
    fpkl.close()

##############################################################
##############################################################
##############################################################




if __name__ == "__main__":
    tracemalloc.start()
    
    ###################
    ### general histos:
    histos = {}
    # hist.book(histos)
    hist.book_minimal(histos) ### only relevant for getting the SMALL_hdL_vs_E and SMALL_E histos
    
    ###################
    ### write the grid:
    save_continuous_grid(histos["SMALL_hdL_vs_E"])
    save_secondaries_grid(histos["SMALL_E"])
    
    ###################################
    ### get the parameters of the model
    dEdxModel  = "G4:Tcut" # or "BB:Tcut"
    TargetMat  = mat.Si # or e.g. mat.Al
    PrimaryPrt = prt.Particle(name="proton",meV=938.27208816*U.MeV2eV,mamu=1.007276466621,chrg=+1.,spin=0.5,lepn=0,magm=2.79284734463)
    par        = flct.Parameters(PrimaryPrt,TargetMat,dEdxModel,"inputs/dEdx_p_si.txt")

    ######################################
    ### claen the scan path from old files
    print("\nClean scan path...")
    ROOT.gSystem.Exec(f"/bin/rm -rf {pklpath}") ## remove old files
    ROOT.gSystem.Exec(f"/bin/mkdir -p {pklpath}")
    
    ################################################
    ### initialize the shapes of all relevant slices
    print("\nSubmit the model jobs...")
    nCPUs = mp.cpu_count() if(parallelize) else 0
    print("nCPUs available:",nCPUs)
    
    
    ###########################################
    ### Create a pool of workers for continuous
    pool = mp.Pool(processes=nCPUs,maxtasksperchild=10) if(parallelize) else None
    for ie in range(1,histos["SMALL_hdL_vs_E"].GetNbinsX()+1):
        for il in range(1,histos["SMALL_hdL_vs_E"].GetNbinsY()+1):
            ### get the slice parameters
            label, E, L = get_continuous_slice(ie,il,histos["SMALL_hdL_vs_E"])
            slice_edges = get_edges_continuous(ie,il,histos["SMALL_hdL_vs_E"])
            P           = par.GetModelPars(E,L)
            print(f'Sending continuous-loss job: label={label}, build={P["build"]}, E={E*U.eV2MeV} MeV, L={L*U.cm2um} um')
            ################
            ### send the job
            if(parallelize):
                pool.apply_async(get_slice_continuous_shapes, args=(E,L,P,label,slice_edges), error_callback=collect_errors)
            else:
                get_slice_continuous_shapes(E,L,P,label,slice_edges)
    ######################################
    ### Wait for all the workers to finish
    if(parallelize): 
        pool.close()
        pool.join()


    ############################################
    ### Create a pool of workers for secondaries
    pool = mp.Pool(processes=nCPUs,maxtasksperchild=10) if(parallelize) else None
    LMID = (histos["SMALL_hdL_vs_E"].GetYaxis().GetXmax()-histos["SMALL_hdL_vs_E"].GetYaxis().GetXmin())/2.
    for ie in range(1,histos["SMALL_E"].GetNbinsX()+1):
        ### get the slice parameters
        label, E    = get_secondaries_slice(ie,histos["SMALL_E"])
        slice_edges = get_edges_secondaries(ie,histos["SMALL_E"])
        P           = par.GetModelPars(E,LMID)
        print(f'Sending secondaries-loss job: label={label}, build={P["build"]}, E={E*U.eV2MeV} MeV, L={L*U.cm2um} um')
        ################
        ### send the job
        if(parallelize):
            pool.apply_async(get_slice_secondaries_shapes, args=(E,LMID,P,label,slice_edges), error_callback=collect_errors)
        else:
            get_slice_secondaries_shapes(E,LMID,P,label,slice_edges)
    ######################################
    ### Wait for all the workers to finish
    if(parallelize): 
        pool.close()
        pool.join()
    
    
    ######################################
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')
    print("[ Top 10 ]")
    for stat in top_stats[:10]:
        print(stat)