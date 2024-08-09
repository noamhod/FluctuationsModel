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

#####################################
### model histos (only relevant ones)
shapes = {}

################################
### the of all pickle files
scanpath = "/Users/noamtalhod/tmp/pkl"


##############################################################
##############################################################
##############################################################
### functions for the submission of model calculation

def get_slice(ie,il,hgrid):
    label_E     = str(ie)
    label_dL    = str(il)
    label       = "E"+label_E+"_dL"+label_dL
    E           = hgrid.GetXaxis().GetBinCenter(ie)*U.MeV2eV # eV
    L           = hgrid.GetYaxis().GetBinCenter(il)*U.um2cm  # cm
    return label, E, L

def add_slice_shapes(E,L,pars,label):
    if(parallelize): 
        lock = mp.Lock()
        lock.acquire()
    start = time.time()
    Mod = model.Model(L,E,pars)
    Mod.set_fft_sampling_pars_rotem(N_t_bins=10000000,frac=0.01)
    Mod.set_all_shapes()
    local_shapes = {label:{"cnt_cdf_arrx":Mod.cnt_cdfs_scaled_arrx,
                           "cnt_cdf_arrsy":Mod.cnt_cdfs_scaled_arrsy,
                           "sec_cdf_arrx":Mod.sec_cdfs_arrx,
                           "sec_cdf_arrsy":Mod.sec_cdfs_arrsy}}
    end = time.time()
    elapsed = end-start
    print(f"Finished slice: {label} at (E,dL)=({E*U.eV2MeV:.3f} MeV,{L*U.cm2um:.6f} um), model shapes obtained within {elapsed:.2f} [s]")
    if(parallelize): lock.release()
    return local_shapes

def collect_errors(error):
    ### https://superfastpython.com/multiprocessing-pool-error-callback-functions-in-python/
    print(f'Error: {error}', flush=True)

def collect_shapes(local_shapes):
    ### https://www.machinelearningplus.com/python/parallel-processing-python/
    global shapes ### defined above!!!
    for label,shape in local_shapes.items(): ### there should be just one item here
        for name,obj in shape.items():
            if(obj is None): continue
            if("arr" in name): shapes[label][name] = obj
            else:
                print("only dealing with arrays. quitting.")
                quit()

        

##############################################################
##############################################################
##############################################################
### functions for the submission of plotting of png's
def save_slice(shapes,builds,label,E,L,P,slice_edges):
    if(parallelize): 
        lock = mp.Lock()
        lock.acquire()
    start = time.time()
    
    ############################
    ### output files
    pklname = f"{scanpath}/slice_{label}.pkl"
    fpkl = open(pklname,"wb")
    ############################

    ### write the sapes to the pickle file
    data = { "Label":label,
             "Slice_E_min":slice_edges["E"][0],
             "Slice_E_max":slice_edges["E"][1],
             "Slice_dL_min":slice_edges["L"][0],
             "Slice_dL_max":slice_edges["L"][1],
             "Pars":P,
             "cnt_cdf_arrx":shapes[label]["cnt_cdf_arrx"],
             "cnt_cdf_arrsy":shapes[label]["cnt_cdf_arrsy"],
             "sec_cdf_arrx":shapes[label]["sec_cdf_arrx"],
             "sec_cdf_arrsy":shapes[label]["sec_cdf_arrsy"] }
    pickle.dump(data, fpkl, protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
    fpkl.close()
    
    end = time.time()
    elapsed = end-start
    print(f"Finished getting slice: {label} with build {builds[label]} within {elapsed:.2f} [s]")
    if(parallelize): lock.release()
    


def save_grid(h2D):
    arrE_mid = np.zeros( h2D.GetNbinsX() )
    arrE_min = np.zeros( h2D.GetNbinsX() )
    arrE_max = np.zeros( h2D.GetNbinsX() )
    for ie in range(1,h2D.GetNbinsX()+1):
        EE   =h2D.GetXaxis().GetBinCenter(ie)
        Emin =h2D.GetXaxis().GetBinLowEdge(ie)
        Emax =h2D.GetXaxis().GetBinUpEdge(ie)
        arrE_mid[ie-1] = EE
        arrE_min[ie-1] = Emin
        arrE_max[ie-1] = Emax
    arrdL_mid = np.zeros( h2D.GetNbinsY() )
    arrdL_min = np.zeros( h2D.GetNbinsY() )
    arrdL_max = np.zeros( h2D.GetNbinsY() )
    for il in range(1,h2D.GetNbinsY()+1):
        LL    = h2D.GetYaxis().GetBinCenter(il)
        dLmin = h2D.GetYaxis().GetBinLowEdge(il)
        dLmax = h2D.GetYaxis().GetBinUpEdge(il)
        arrdL_mid[il-1] = LL
        arrdL_min[il-1] = dLmin
        arrdL_max[il-1] = dLmax
    slice_arrays = {"arrE":arrE_mid,"arrE_min":arrE_min,"arrE_max":arrE_max, "arrdL":arrdL_mid,"arrdL_min":arrdL_min,"arrdL_max":arrdL_max}
    fpkl = open("scan_grid.pkl","wb")
    pickle.dump(slice_arrays, fpkl, protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
    fpkl.close()

##############################################################
##############################################################
##############################################################




if __name__ == "__main__":
    ###################
    ### general histos:
    histos = {}
    hist.book(histos) ### only relevant for getting the SMALL_hdL_vs_E hist
    
    ###################
    ### write the grid:
    save_grid(histos["SMALL_hdL_vs_E"])
    
    ###################################
    ### get the parameters of the model
    dEdxModel  = "G4:Tcut" # or "BB:Tcut"
    TargetMat  = mat.Si # or e.g. mat.Al
    PrimaryPrt = prt.Particle(name="proton",meV=938.27208816*U.MeV2eV,mamu=1.007276466621,chrg=+1.,spin=0.5,lepn=0,magm=2.79284734463)
    par        = flct.Parameters(PrimaryPrt,TargetMat,dEdxModel,"inputs/dEdx_p_si.txt")

    ######################################
    ### claen the scan path from old files
    print("\nClean scan path...")
    ROOT.gSystem.Exec(f"/bin/mkdir -p {scanpath}")
    
    ################################################
    ### initialize the shapes of all relevant slices
    print(f"\nBook shapes...")
    for ie in range(1,histos["SMALL_hdL_vs_E"].GetNbinsX()+1):
        for il in range(1,histos["SMALL_hdL_vs_E"].GetNbinsY()+1):
            ### get the slice parameters
            label, E, L = get_slice(ie,il,histos["SMALL_hdL_vs_E"])
            ### init the relevant model shapes
            shapes.update( {label : {"E":E, "L":L} } )

    #############################################
    ### collect the shapes of all relevant slices
    print("\nSubmit the model jobs...")
    nCPUs = mp.cpu_count() if(parallelize) else 0
    print("nCPUs available:",nCPUs)
    ### Create a pool of workers
    pool = mp.Pool(nCPUs) if(parallelize) else None
    builds = {}
    for label,shape in shapes.items():
        E = shape["E"]
        L = shape["L"]
        P = par.GetModelPars(E,L)
        builds.update({label:P["build"]})
        print(f'Sending job: label={label}, build={P["build"]}, E={E*U.eV2MeV} MeV, L={L*U.cm2um} um')
        ########################
        ### send the job
        if(parallelize):
            pool.apply_async(add_slice_shapes, args=(E,L,P,label), callback=collect_shapes, error_callback=collect_errors)
        else:
            local_shapes = add_slice_shapes(E,L,P,label)
            collect_shapes(local_shapes)        
    ######################################
    ### Wait for all the workers to finish
    if(parallelize): 
        pool.close()
        pool.join()

    ###################
    ### save all slices
    print(f"\nSaving slices...")
    parallelize = False
    pool = mp.Pool(nCPUs) if(parallelize) else None
    for ie in range(1,histos["SMALL_hdL_vs_E"].GetNbinsX()+1):
        for il in range(1,histos["SMALL_hdL_vs_E"].GetNbinsY()+1):
            ### get the slice parameters
            label, E, L = get_slice(ie,il,histos["SMALL_hdL_vs_E"])
            Emin        = histos["SMALL_hdL_vs_E"].GetXaxis().GetBinLowEdge(ie)
            Emax        = histos["SMALL_hdL_vs_E"].GetXaxis().GetBinUpEdge(ie)
            Lmin        = histos["SMALL_hdL_vs_E"].GetYaxis().GetBinLowEdge(il)
            Lmax        = histos["SMALL_hdL_vs_E"].GetYaxis().GetBinUpEdge(il)
            slice_edges = {"E":[Emin,Emax], "L":[Lmin,Lmax]}
            P           = par.GetModelPars(E,L)
            ### send the job
            if(parallelize):
                pool.apply_async(save_slice, args=(shapes,builds,label,E,L,P,slice_edges), error_callback=collect_errors)
            else:
                save_slice(shapes,builds,label,E,L,P,slice_edges)
    ######################################
    ### Wait for all the workers to finish
    if(parallelize): 
        pool.close()
        pool.join()