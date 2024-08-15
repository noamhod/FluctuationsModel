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

import argparse

# parser = argparse.ArgumentParser(description='scan_example.py...')
# parser.add_argument('-N', metavar='N steps to process if less than all, for all put 0', required=True,  help='N steps to process if les than all, for all put 0')
# argus = parser.parse_args()
NN = 0#int(argus.N)

ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.15)


### model shapes defined later as global
parallelize = False

#####################################
### model histos (only relevant ones)
shapes = {}

#################################
### slice histos with the MC data
slices = {}

################################
### the of all slices
rootpath = "./output"


##############################################################
##############################################################
##############################################################
### functions for the submission of model calculation

def get_slice(ie,il):
    label_E   = str(ie)
    label_dL  = str(il)
    label     = "E"+label_E+"_dL"+label_dL
    NrawSteps = histos["SMALL_hdL_vs_E"].GetBinContent(ie,il)
    midRangeE = slices["hE_"+label].GetXaxis().GetXmin()  + (slices["hE_"+label].GetXaxis().GetXmax()-slices["hE_"+label].GetXaxis().GetXmin())/2.
    midRangeL = slices["hdL_"+label].GetXaxis().GetXmin() + (slices["hdL_"+label].GetXaxis().GetXmax()-slices["hdL_"+label].GetXaxis().GetXmin())/2.
    E = midRangeE*U.MeV2eV # eV
    L = midRangeL*U.um2cm  # cm
    return label, E, L, NrawSteps

def add_slice_shapes(E,L,pars,N,label):
    if(parallelize): 
        lock = mp.Lock()
        lock.acquire()
    start = time.time()
    Mod = model.Model(L,E,pars)
    # Mod.set_fft_sampling_pars(N_t_bins=10000000,frac=0.01)
    Mod.set_fft_sampling_pars_rotem(N_t_bins=10_000_000,frac=0.01)
    Mod.set_all_shapes()
    local_shapes = {label:{"cnt_pdf":Mod.cnt_pdfs_scaled["hModel"],
                           "cnt_pdf_all":Mod.cnt_pdfs_scaled,
                           "sec_pdf":Mod.sec_pdfs["hBorysov_Sec"],
                           "cnt_cdf":Mod.cnt_cdfs_scaled["hModel"],
                           "sec_cdf":Mod.sec_cdfs["hBorysov_Sec"],
                           "cnt_cdf_arrx":Mod.cnt_cdfs_scaled_arrx,
                           "cnt_cdf_arrsy":Mod.cnt_cdfs_scaled_arrsy,
                           "sec_cdf_arrx":Mod.sec_cdfs_arrx,
                           "sec_cdf_arrsy":Mod.sec_cdfs_arrsy}}
    end = time.time()
    elapsed = end-start
    print(f"Finished slice: {label} with {int(N):,} steps, at (E,dL)=({E*U.eV2MeV:.3f} MeV,{L*U.cm2um:.6f} um), model shapes obtained within {elapsed:.2f} [s]")
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
                if(name=="cnt_pdf_all"):
                    for componentname,componenthist in local_shapes[label][name].items():
                        shapes[label][name].update({componentname : componenthist.Clone(label+"_"+componentname) })
                else:
                    shapes[label][name] = obj.Clone(label+"_"+name)

        

##############################################################
##############################################################
##############################################################
### functions for the submission of plotting of png's
def save_slice(slices,shapes,builds,label,E,L,P,NrawSteps,count):
    if(parallelize): 
        lock = mp.Lock()
        lock.acquire()
    start = time.time()
    
    ### get the precalculated model shapes
    ### this is not saved in the root file so defined above it
    cnt_pdf_all = shapes[label]["cnt_pdf_all"]
    cnt_pdf = shapes[label]["cnt_pdf"]
    sec_pdf = shapes[label]["sec_pdf"]
    cnt_cdf = shapes[label]["cnt_cdf"]
    sec_cdf = shapes[label]["sec_cdf"]
    
    ############################
    ### ROOT file for the output
    tfname = f"{rootpath}/rootslice_{label}.root"
    pklname = f"{rootpath}/rootslice_{label}.pkl"
    tf = ROOT.TFile(tfname,"RECREATE")
    fpkl = open(pklname,"wb")
    tf.cd()
    ############################
    
    build = ROOT.TNamed("build", builds[label])
    build.Write()
    
    slices["hdEcnt_"+label].Write()
    slices["hdEsec_"+label].Write()
    slices["hE_"+label].Write()
    slices["hdL_"+label].Write()
    # cnt_slice_cdf.Write()
    # sec_slice_cdf.Write()
    if(cnt_pdf is not None): cnt_pdf.Write()
    if(sec_pdf is not None): sec_pdf.Write()
    if(cnt_cdf is not None): cnt_cdf.Write()
    if(sec_cdf is not None): sec_cdf.Write()
    
    tree = ROOT.TTree("meta_"+label,"meta_"+label)
    kstest_prob_cnt_pdf = array.array('d',[0])
    kstest_prob_sec_pdf = array.array('d',[0])
    kstest_dist_cnt_pdf = array.array('d',[0])
    kstest_dist_sec_pdf = array.array('d',[0])
    c2test_ndof_cnt_pdf = array.array('d',[0])
    c2test_ndof_sec_pdf = array.array('d',[0])
    
    tree.Branch('KS_prob_test_cnt_pdf',kstest_prob_cnt_pdf,'KS_prob_test_cnt_pdf/D')
    tree.Branch('KS_prob_test_sec_pdf',kstest_prob_sec_pdf,'KS_prob_test_sec_pdf/D')
    tree.Branch('KS_dist_test_cnt_pdf',kstest_dist_cnt_pdf,'KS_dist_test_cnt_pdf/D')
    tree.Branch('KS_dist_test_sec_pdf',kstest_dist_sec_pdf,'KS_dist_test_sec_pdf/D')
    tree.Branch('C2_ndof_test_cnt_pdf',c2test_ndof_cnt_pdf,'C2_ndof_test_cnt_pdf/D')
    tree.Branch('C2_ndof_test_sec_pdf',c2test_ndof_sec_pdf,'C2_ndof_test_sec_pdf/D')
    
    kstest_prob_cnt_pdf[0] = -1
    kstest_dist_cnt_pdf[0] = -1
    c2test_ndof_cnt_pdf[0] = -1
    kstest_prob_sec_pdf[0] = -1
    kstest_dist_sec_pdf[0] = -1
    c2test_ndof_sec_pdf[0] = -1
    
    if(cnt_pdf is not None and slices["hdEcnt_"+label].Integral()>0):# and cnt_pdf.Integral()>0):
        kstest_prob_cnt_pdf[0] = slices["hdEcnt_"+label].KolmogorovTest(cnt_pdf)
        kstest_dist_cnt_pdf[0] = slices["hdEcnt_"+label].KolmogorovTest(cnt_pdf,"M")
        c2test_ndof_cnt_pdf[0] = slices["hdEcnt_"+label].Chi2Test(cnt_pdf,"CHI2/NDF")
    if(sec_pdf is not None and slices["hdEsec_"+label].Integral()>0):# and sec_pdf.Integral()>0):
        kstest_prob_sec_pdf[0] = slices["hdEsec_"+label].KolmogorovTest(sec_pdf)
        kstest_dist_sec_pdf[0] = slices["hdEsec_"+label].KolmogorovTest(sec_pdf,"M")
        c2test_ndof_sec_pdf[0] = slices["hdEsec_"+label].Chi2Test(sec_pdf,"CHI2/NDF")
    tree.Fill()
    # tree.Write()
    tf.Write()
    tf.Close()
    
    ### write the sapes to the pickle file
    data = { "Label":label,
             "Slice_E_min":slices["hE_"+label].GetXaxis().GetXmin(),
             "Slice_E_max":slices["hE_"+label].GetXaxis().GetXmax(),
             "Slice_dL_min":slices["hdL_"+label].GetXaxis().GetXmin(),
             "Slice_dL_max":slices["hdL_"+label].GetXaxis().GetXmax(),
             "Pars":P,
             "cnt_cdf_arrx":shapes[label]["cnt_cdf_arrx"],
             "cnt_cdf_arrsy":shapes[label]["cnt_cdf_arrsy"],
             "sec_cdf_arrx":shapes[label]["sec_cdf_arrx"],
             "sec_cdf_arrsy":shapes[label]["sec_cdf_arrsy"] }
    pickle.dump(data, fpkl, protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
    fpkl.close()
    
    end = time.time()
    elapsed = end-start
    print(f"Finished plotting slice: {label} with build {builds[label]} with KSprob={kstest_prob_cnt_pdf[0]}, within {elapsed:.2f} [s]")
    if(parallelize): lock.release()
    

##############################################################
##############################################################
##############################################################




if __name__ == "__main__":
    # open a file, where you stored the pickled data
    # fileY = open('data/with_secondaries/step_info_df_no_msc.pkl', 'rb')
    use_data = False
    if use_data:
        fileY = open('data/steps_info_18_07_2024.pkl', 'rb')
        # dump information to that file
        Y = pickle.load(fileY)
        # close the file
        fileY.close()
        df = pd.DataFrame(Y)
        # print(df)
        arr_dx     = df['dX'].to_numpy()
        arr_dy     = df['dY'].to_numpy()
        arr_dz     = df['dZ'].to_numpy()
        arr_dEcnt  = df['ContinuousLoss'].to_numpy()
        arr_dEtot  = df['TotalEnergyLoss'].to_numpy()
        arr_E      = df['KineticEnergy'].to_numpy()
        arr_dR     = df['dR'].to_numpy()
        arr_dL     = df['dL'].to_numpy()
        arr_dTheta = df['dTheta'].to_numpy()
        arr_dPhi   = df['dPhi'].to_numpy()


    ###################
    ### general histos:
    histos = {}
    hist.book(histos)
    
    arrE_mid = np.zeros( histos["SMALL_hdL_vs_E"].GetNbinsX() )
    arrE_min = np.zeros( histos["SMALL_hdL_vs_E"].GetNbinsX() )
    arrE_max = np.zeros( histos["SMALL_hdL_vs_E"].GetNbinsX() )
    for ie in range(1,histos["SMALL_hdL_vs_E"].GetNbinsX()+1):
        EE   = histos["SMALL_hdL_vs_E"].GetXaxis().GetBinCenter(ie)
        Emin = histos["SMALL_hdL_vs_E"].GetXaxis().GetBinLowEdge(ie)
        Emax = histos["SMALL_hdL_vs_E"].GetXaxis().GetBinUpEdge(ie)
        arrE_mid[ie-1] = EE
        arrE_min[ie-1] = Emin
        arrE_max[ie-1] = Emax
    arrdL_mid = np.zeros( histos["SMALL_hdL_vs_E"].GetNbinsY() )
    arrdL_min = np.zeros( histos["SMALL_hdL_vs_E"].GetNbinsY() )
    arrdL_max = np.zeros( histos["SMALL_hdL_vs_E"].GetNbinsY() )
    for il in range(1,histos["SMALL_hdL_vs_E"].GetNbinsY()+1):
        LL    = histos["SMALL_hdL_vs_E"].GetYaxis().GetBinCenter(il)
        dLmin = histos["SMALL_hdL_vs_E"].GetYaxis().GetBinLowEdge(il)
        dLmax = histos["SMALL_hdL_vs_E"].GetYaxis().GetBinUpEdge(il)
        arrdL_mid[il-1] = LL
        arrdL_min[il-1] = dLmin
        arrdL_max[il-1] = dLmax
    slice_arrays = {"arrE":arrE_mid,"arrE_min":arrE_min,"arrE_max":arrE_max, "arrdL":arrdL_mid,"arrdL_min":arrdL_min,"arrdL_max":arrdL_max}
    fpkl = open("scan_example.pkl","wb")
    pickle.dump(slice_arrays, fpkl, protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
    fpkl.close()
    
    
    ###################################
    ### get the parameters of the model
    dEdxModel  = "G4:Tcut" # or "BB:Tcut"
    TargetMat  = mat.Si # or e.g. mat.Al
    PrimaryPrt = prt.Particle(name="proton",meV=938.27208816*U.MeV2eV,mamu=1.007276466621,chrg=+1.,spin=0.5,lepn=0,magm=2.79284734463)
    par        = flct.Parameters(PrimaryPrt,TargetMat,dEdxModel,"inputs/dEdx_p_si.txt")
    
    #####################################################
    ### first define the slice histos to hold the MC data
    print("\nDefine slices with the proper binning as determined by the model...")
    for ie in range(1,histos["SMALL_hdL_vs_E"].GetNbinsX()+1):
        label_E = str(ie)
        EE   = histos["SMALL_hdL_vs_E"].GetXaxis().GetBinCenter(ie)
        Emin = histos["SMALL_hdL_vs_E"].GetXaxis().GetBinLowEdge(ie)
        Emax = histos["SMALL_hdL_vs_E"].GetXaxis().GetBinUpEdge(ie)
        for il in range(1,histos["SMALL_hdL_vs_E"].GetNbinsY()+1):
            label_dL = str(il)
            LL    = histos["SMALL_hdL_vs_E"].GetYaxis().GetBinCenter(il)
            dLmin = histos["SMALL_hdL_vs_E"].GetYaxis().GetBinLowEdge(il)
            dLmax = histos["SMALL_hdL_vs_E"].GetYaxis().GetBinUpEdge(il)
            label = "E"+label_E+"_dL"+label_dL
            #######################################################
            ### find the parameters (mostly histos limits and bins)
            modelpars = par.GetModelPars(EE*U.MeV2eV,LL*U.um2cm)
            Mod = model.Model(LL*U.um2cm, EE*U.MeV2eV, modelpars)
            ############################################
            ### now define the histos of the the MC data - the binning changes according to the model!
            slices.update({"hE_"+label:  ROOT.TH1D("hE_"+label,label+";E [MeV];Steps", bins.n_E,Emin,Emax)})
            slices.update({"hdL_"+label: ROOT.TH1D("hdL_"+label,label+";#DeltaL [#mum];Steps", int(bins.n_dL/10),dLmin,dLmax)})
            slices.update({"hdEcnt_"+label: ROOT.TH1D("hdEcnt_"+label,label+";#DeltaE [eV];Steps", Mod.NbinsScl,Mod.dEminScl,Mod.dEmaxScl)})
            slices.update({"hdEsec_"+label: ROOT.TH1D("hdEsec_"+label,label+";#DeltaE [eV];Steps", Mod.NbinsSec,Mod.dEminSec,Mod.dEmaxSec)})
                

    if use_data:
        #######################################
        ### Run the MC data and fill the histos
        print("\nStart the loop over GEANT4 data...")
        for n in range(len(arr_dx)):
            dx     = arr_dx[n]*U.m2um
            dxinv  = 1/dx if(dx>0) else -999
            dy     = arr_dy[n]*U.m2um
            dz     = arr_dz[n]*U.m2um
            dEcnt  = arr_dEcnt[n]*U.eV2MeV
            dEtot  = arr_dEtot[n]*U.eV2MeV
            dEsec  = dEtot - dEcnt
            dE     = dEtot
            E      = arr_E[n]*U.eV2MeV
            dR     = arr_dR[n]*U.m2um
            dRinv = 1/dR if(dR>0) else -999 ## this happens for the primary particles...
            dL     = arr_dL[n]*U.m2um
            dTheta = arr_dTheta[n]
            dPhi   = arr_dPhi[n]

            ################
            ### valideations
            if(E>=bins.Emax):   continue ## skip the primary particles
            if(E<bins.Emin):    continue ## skip the low energy particles
            if(dx>=bins.dxmax): continue ## skip
            if(dx<bins.dxmin):  continue ## skip

            histos["hE"].Fill(E)
            histos["hdE"].Fill(dE)
            histos["hdx"].Fill(dx)
            if(dx>0): histos["hdxinv"].Fill(dxinv)
            histos["hdR"].Fill(dR)
            histos["hdL"].Fill(dL)
            histos["hdRinv"].Fill(dRinv)
            if(dx>0): histos["hdEdx"].Fill(dE/dx)
            if(dx>0): histos["hdEdx_vs_E"].Fill(E,dE/dx)
            histos["hdE_vs_dx"].Fill(dx,dE)
            if(dx>0): histos["hdE_vs_dxinv"].Fill(dxinv,dE)
            histos["hdx_vs_E"].Fill(E,dx)
            if(dx>0): histos["hdxinv_vs_E"].Fill(E,dxinv)
            histos["hdL_vs_E"].Fill(E,dL)
            histos["SMALL_hdL_vs_E"].Fill(E,dL)

            ie = histos["SMALL_hdL_vs_E"].GetXaxis().FindBin(E)
            il = histos["SMALL_hdL_vs_E"].GetYaxis().FindBin(dx)
            label = "E"+str(ie)+"_dL"+str(il)
            slices["hdEcnt_"+label].Fill(dEcnt*U.MeV2eV)
            slices["hdEsec_"+label].Fill(dEsec*U.MeV2eV)
            slices["hE_"+label].Fill(E)
            slices["hdL_"+label].Fill(dL)

            if(n%1000000==0 and n>0): print("processed: ",n)
            if(n>NN and NN>0): break
    

    ##########################################################
    ### plot the basic diagnostics histos (not yet the slices)
    print("\nPlot some basic histograms...")
    pdf = "scan_example.pdf"
    cnv = ROOT.TCanvas("cnv","",1000,1000)
    cnv.Divide(2,2)
    cnv.cd(1)
    histos["hE"].Draw("hist")
    histos["hE"].SetMinimum(1)
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    histos["hdE"].Draw("hist")
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.cd(3)
    histos["hdx"].Draw("hist")
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.cd(4)
    histos["hdL"].Draw("hist")
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.SaveAs(pdf+"(")
    #####################
    cnv = ROOT.TCanvas("cnv","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    histos["hdEdx"].Draw("hist")
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    histos["hdEdx_vs_E"].Draw("colz")
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.SaveAs(pdf)
    #####################
    cnv = ROOT.TCanvas("cnv","",1000,500)
    cnv.Divide(2,1)
    cnv.cd(1)
    histos["hdE_vs_dx"].Draw("colz")
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.cd(2)
    histos["hdE_vs_dxinv"].Draw("colz")
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.SaveAs(pdf)
    #####################    
    cnv = ROOT.TCanvas("cnv","",500,500)
    histos["hdL_vs_E"].Draw("colz")
    gridx,gridy = hist.getGrid(histos["SMALL_hdL_vs_E"])
    for line in gridx:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    for line in gridy:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.SaveAs(pdf)
    #####################    
    cnv = ROOT.TCanvas("cnv","",500,500)
    histos["SMALL_hdL_vs_E"].Draw("colz")
    gridx,gridy = hist.getGrid(histos["SMALL_hdL_vs_E"])
    for line in gridx:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    for line in gridy:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.SaveAs(pdf+")")
    
    ###################################
    ### write everything to a root file
    print("\nWriting root file...")
    tfout = ROOT.TFile("scan_example.root","RECREATE")
    tfout.cd()
    # for name,h in histos.items(): h.Write()
    # for name,h in slices.items(): h.Write()
    histos["hE"].Write()
    histos["hdE"].Write()
    histos["hdx"].Write()
    histos["hdxinv"].Write()
    histos["hdR"].Write()
    histos["hdL"].Write()
    histos["hdRinv"].Write()
    histos["hdEdx"].Write()
    histos["hdEdx_vs_E"].Write()
    histos["hdE_vs_dx"].Write()
    histos["hdE_vs_dxinv"].Write()
    histos["hdx_vs_E"].Write()
    histos["hdxinv_vs_E"].Write()
    histos["hdL_vs_E"].Write()
    histos["SMALL_hdL_vs_E"].Write()
    tfout.Write()
    tfout.Close()

    #######################################################################
    #######################################################################
    #######################################################################
    
    #########################
    ### make gif for all bins
    print("\nClean temp png's and temp png path...")
    ROOT.gSystem.Exec(f"/bin/rm -rf {rootpath}") ## remove old files
    ROOT.gSystem.Exec(f"/bin/mkdir -p {rootpath}")
    
    ################################################
    ### initialize the shapes of all relevant slices
    print(f"\nBook shapes...")
    NrawStepsIgnore = 0 #1 #10
    for ie in range(1,histos["SMALL_hdL_vs_E"].GetNbinsX()+1):
        for il in range(1,histos["SMALL_hdL_vs_E"].GetNbinsY()+1):
            ### get the slice parameters
            label, E, L, NrawSteps = get_slice(ie,il)
            ### skip if too few entries
            if(NrawSteps<NrawStepsIgnore): continue
            ### init the relevant model shapes
            shapes.update( {label : {"E":E, "L":L, "N":NrawSteps, "cnt_pdf":None, "cnt_pdf_all":{}, "sec_pdf":None, "cnt_cdf":None, "sec_cdf":None} } )
    

    #############################################
    ### collect the shapes of all relevant slices
    print("\nSubmit the model jobs...")
    nCPUs = mp.cpu_count() if(parallelize) else 0
    print("nCPUs available:",nCPUs)
    ### Create a pool of workers
    pool = mp.Pool(processes=nCPUs,maxtasksperchild=10) if(parallelize) else None
    builds = {}
    for label,shape in shapes.items():
        E = shape["E"]
        L = shape["L"]
        N = shape["N"]
        P = par.GetModelPars(E,L)
        builds.update({label:P["build"]})
        print(f'Sending job: label={label}, build={P["build"]}, E={E*U.eV2MeV} MeV, L={L*U.cm2um} um, N={N} steps')
        ########################
        ### get the model shapes
        if(parallelize):
            pool.apply_async(add_slice_shapes, args=(E,L,P,N,label), callback=collect_shapes, error_callback=collect_errors)
        else:
            local_shapes = add_slice_shapes(E,L,P,N,label)
            collect_shapes(local_shapes)        
    ######################################
    ### Wait for all the workers to finish
    if(parallelize): 
        pool.close()
        pool.join()

    #######################################################################
    #######################################################################
    #######################################################################

    #############################################
    ### post processing: plot the relevant slices
    print("\nPlot all slices against the model shapes...")
    parallelize = False
    print(f"\nPlotting shapes... (with parallelize={parallelize})")
    count = 0
    pool = mp.Pool(nCPUs) if(parallelize) else None
    for ie in range(1,histos["SMALL_hdL_vs_E"].GetNbinsX()+1):
        for il in range(1,histos["SMALL_hdL_vs_E"].GetNbinsY()+1):
            ### get the slice parameters
            label, E, L, NrawSteps = get_slice(ie,il)
            P = par.GetModelPars(E,L)
            ### skip if too few entries
            if(NrawSteps<NrawStepsIgnore): continue
            if(parallelize):
                pool.apply_async(save_slice, args=(slices,shapes,builds,label,E,L,P,NrawSteps,count), error_callback=collect_errors)
            else:
                save_slice(slices,shapes,builds,label,E,L,P,NrawSteps,count)
            count += 1
    ######################################
    ### Wait for all the workers to finish
    if(parallelize): 
        pool.close()
        pool.join()
