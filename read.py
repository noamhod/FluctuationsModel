import time
import pickle
import math
import array
import numpy as np
import ROOT
import units as U
import constants as C
import material as mat
import bins
import fluctuations as flct
import shapes
import hist
import model
import multiprocessing as mp

import argparse
parser = argparse.ArgumentParser(description='readtest.py...')
parser.add_argument('-N', metavar='N steps to process if less than all, for all put 0', required=True,  help='N steps to process if les than all, for all put 0')
argus = parser.parse_args()
NN = int(argus.N)

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

#################################
### slice histos with the MC data
slices = {}


def get_slice(ie,ixinv):
    NminRawSteps = 25
    label_E      = str(ie)
    label_dxinv  = str(ixinv)
    label        = "E"+label_E+"_dxinv"+label_dxinv
    NrawSteps = histos["SMALL_hdxinv_vs_E"].GetBinContent(ie,ixinv)
    midRangeE = (slices["hE_"+label].GetXaxis().GetXmax()-slices["hE_"+label].GetXaxis().GetXmin())/2.
    midRangeX = (slices["hdx_"+label].GetXaxis().GetXmax()-slices["hdx_"+label].GetXaxis().GetXmin())/2.
    E = slices["hE_"+label].GetMean()*U.MeV2eV if(NrawSteps>=NminRawSteps) else midRangeE*U.MeV2eV # eV
    x = slices["hdx_"+label].GetMean()*U.um2cm if(NrawSteps>=NminRawSteps) else midRangeX*U.um2cm  # cm
    return label, E, x, NrawSteps

def add_slice_shapes(E,x,pars,N,label):
    if(parallelize): 
        lock = mp.Lock()
        lock.acquire()
    start = time.time()
    Mod = model.Model(x,E,pars)
    Mod.set_fft_sampling_pars(N_t_bins=10000000,frac=0.01)
    Mod.set_all_shapes()
    local_shapes = {label:{"cnt_pdf":Mod.cnt_pdfs_scaled["hModel"], "sec_pdf":Mod.sec_pdfs["hBorysov_Sec"], "cnt_cdf":Mod.cnt_cdfs_scaled["hModel"], "sec_cdf":Mod.sec_cdfs["hBorysov_Sec"]}}
    end = time.time()
    elapsed = end-start
    print(f"Finished slice: {label} with {int(N):,} steps, at (E,dx)=({E*U.eV2MeV:.3f} MeV,{x*U.cm2um:.6f} um), model shapes obtained within {elapsed:.2f} [s]")
    if(parallelize): lock.release()
    return local_shapes

def collect_errors(error):
    ### https://superfastpython.com/multiprocessing-pool-error-callback-functions-in-python/
    print(f'Error: {error}', flush=True)

def collect_shapes(local_shapes):
    ### https://www.machinelearningplus.com/python/parallel-processing-python/
    global shapes ### defined above!!!
    for label,shape in local_shapes.items(): ### there should be just one item here
        for name,hist in shape.items():
            if(hist is None): continue
            shapes[label][name] = hist.Clone(label+"_"+name)




if __name__ == "__main__":
    ##################################################
    ### open a file, where you stored the pickled data
    fileX = open('data/without_E2MeV_cut/with_multiple_scattering/X.pkl', 'rb')
    fileY = open('data/without_E2MeV_cut/with_multiple_scattering/Y.pkl', 'rb')
    #################################
    ### dump information to that file
    X = pickle.load(fileX)
    Y = pickle.load(fileY)
    ##################
    ### close the file
    fileX.close()
    fileY.close()
    ###################
    ### general histos:
    histos = {}
    hist.book(histos)
    
    ###################################
    ### get the parameters of the model
    TargetMat = mat.Si # or e.g. mat.Al
    ParticleN = "Proton"
    ParticleM = C.mp
    ParticleQ = +1
    ParamName = ParticleN+"_on_"+TargetMat.name
    dEdxModel = "G4:Tcut" # or "BB:Tcut"
    par       = flct.Parameters(ParamName,ParticleM,ParticleQ,TargetMat,dEdxModel,"inputs/eloss_p_si.txt","inputs/BB.csv")
    
    #####################################################
    ### first define the slice histos to hold the MC data
    for ie in range(1,histos["SMALL_hdxinv_vs_E"].GetNbinsX()+1):
        label_E = str(ie)
        EE   = histos["SMALL_hdxinv_vs_E"].GetXaxis().GetBinCenter(ie)
        Emin = histos["SMALL_hdxinv_vs_E"].GetXaxis().GetBinLowEdge(ie)
        Emax = histos["SMALL_hdxinv_vs_E"].GetXaxis().GetBinUpEdge(ie)
        for ixinv in range(1,histos["SMALL_hdxinv_vs_E"].GetNbinsY()+1):
            label_dxinv = str(ixinv)
            XX       = 1./histos["SMALL_hdxinv_vs_E"].GetYaxis().GetBinCenter(ixinv)
            dxinvmin = histos["SMALL_hdxinv_vs_E"].GetYaxis().GetBinLowEdge(ixinv)
            dxinvmax = histos["SMALL_hdxinv_vs_E"].GetYaxis().GetBinUpEdge(ixinv)
            label = "E"+label_E+"_dxinv"+label_dxinv
            #######################################################
            ### find the parameters (mostly histos limits and bins)
            modelpars = par.GetModelPars(EE*U.MeV2eV,XX*U.um2cm)
            Mod = model.Model(XX*U.um2cm, EE*U.MeV2eV, modelpars)
            ############################################
            ### now define the histos of the the MC data
            slices.update({"hE_"+label:  ROOT.TH1D("hE_"+label,label+";E [MeV];Steps", bins.n_small_E,Emin,Emax)})
            slices.update({"hdxinv_"+label: ROOT.TH1D("hdxinv_"+label,label+";1/dx [1/#mum];Steps", bins.n_small_dxinv,dxinvmin,dxinvmax)})
            slices.update({"hdx_"+label: ROOT.TH1D("hdx_"+label,label+";dx [#mum];Steps", bins.n_small_dx,1./dxinvmax,1./dxinvmin)})
            slices.update({"hdEcnt_"+label: ROOT.TH1D("hdEcnt_"+label,label+";#DeltaE [MeV];Steps", Mod.Nbins,Mod.dEmin,Mod.dEmax)})
            slices.update({"hdEsec_"+label: ROOT.TH1D("hdEsec_"+label,label+";#DeltaE [MeV];Steps", Mod.NbinsSec,Mod.dEminSec,Mod.dEmaxSec)})

    #######################################
    ### Run the MC data and fill the histos
    for n,enrgy in enumerate(X):
        E     = enrgy*U.eV2MeV
        dx    = Y[n][0]*U.m2um
        dxinv = 1/dx if(dx>0) else -999
        dR    = Y[n][1]*U.m2um
        dRinv = 1/dR if(dR>0) else -999 ## this happens for the primary particles...
        dEcnt = Y[n][2]*U.eV2MeV
        # dEtot = Y[n][3]*U.eV2MeV
        # dEsec = dEtot-dEcnt
        dEsec = Y[n][3]*U.eV2MeV
        dEtot = dEcnt+dEsec
        dE    = dEtot
        ################
        ### valideations
        if(E>=bins.Emax):   continue ## skip the primary particles
        if(E<bins.Emin):    continue ## skip the low energy particles
        if(dx>=bins.dxmax): continue ## skip
        if(dx<bins.dxmin):  continue ## skip
        
        histos["hE"].Fill(E)
        histos["hdE"].Fill(dE)
        histos["hdx"].Fill(dx)
        histos["hdxinv"].Fill(dxinv)
        histos["hdR"].Fill(dR)
        histos["hdRinv"].Fill(dRinv)
        histos["hdEdx"].Fill(dE/dx)
        histos["hdEdx_vs_E"].Fill(E,dE/dx)
        histos["hdE_vs_dx"].Fill(dx,dE)
        histos["hdE_vs_dxinv"].Fill(dxinv,dE)
        histos["hdx_vs_E"].Fill(E,dx)
        histos["hdxinv_vs_E"].Fill(E,dxinv)
        histos["SMALL_hdx_vs_E"].Fill(E,dx)
        histos["SMALL_hdxinv_vs_E"].Fill(E,dxinv)
        
        # ie = histos["SMALL_hdx_vs_E"].GetXaxis().FindBin(E)
        # ix = histos["SMALL_hdx_vs_E"].GetYaxis().FindBin(dx)
        # label = "E"+str(ie)+"_dx"+str(ix)
        # histos["hdE_"+label].Fill(dE)
        # histos["hE_"+label].Fill(E)
        # histos["hdx_"+label].Fill(dx)
        
        ie    = histos["SMALL_hdxinv_vs_E"].GetXaxis().FindBin(E)
        ixinv = histos["SMALL_hdxinv_vs_E"].GetYaxis().FindBin(dxinv)
        label = "E"+str(ie)+"_dxinv"+str(ixinv)
        slices["hdEcnt_"+label].Fill(dEcnt*U.MeV2eV)
        slices["hdEsec_"+label].Fill(dEsec*U.MeV2eV)
        slices["hE_"+label].Fill(E)
        slices["hdxinv_"+label].Fill(dxinv)
        slices["hdx_"+label].Fill(dx)
    
        if(n%1000000==0 and n>0): print("processed: ",n)
        if(n>NN and NN>0): break
    

    ##########################################################
    ### plot the basic diagnostics histos (not yet the slices)
    pdf = "scan.pdf"
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
    histos["hdR"].Draw("hist")
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.SaveAs(pdf+"(")
    #####################
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
    histos["hdxinv"].Draw("hist")
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.cd(4)
    histos["hdRinv"].Draw("hist")
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.SaveAs(pdf)
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
    histos["hdx_vs_E"].Draw("colz")
    gridx,gridy = hist.getGrid(histos["SMALL_hdx_vs_E"])
    for line in gridx:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    for line in gridy:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    histos["SMALL_hdx_vs_E"].SetMarkerSize(0.2)
    histos["SMALL_hdx_vs_E"].Draw("text same")
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.SaveAs(pdf)
    #####################    
    cnv = ROOT.TCanvas("cnv","",500,500)
    histos["SMALL_hdx_vs_E"].SetMarkerSize(0.2)
    histos["SMALL_hdx_vs_E"].Draw("colz text")
    gridx,gridy = hist.getGrid(histos["SMALL_hdx_vs_E"])
    for line in gridx:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    for line in gridy:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.SaveAs(pdf)
    #####################    
    cnv = ROOT.TCanvas("cnv","",500,500)
    histos["hdxinv_vs_E"].Draw("colz")
    gridx,gridy = hist.getGrid(histos["SMALL_hdxinv_vs_E"])
    for line in gridx:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    for line in gridy:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    histos["SMALL_hdxinv_vs_E"].SetMarkerSize(0.2)
    histos["SMALL_hdxinv_vs_E"].Draw("text same")
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.SaveAs(pdf)
    #####################    
    cnv = ROOT.TCanvas("cnv","",500,500)
    histos["SMALL_hdxinv_vs_E"].SetMarkerSize(0.2)
    histos["SMALL_hdxinv_vs_E"].Draw("colz text")
    gridx,gridy = hist.getGrid(histos["SMALL_hdxinv_vs_E"])
    for line in gridx:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    for line in gridy:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.SaveAs(pdf+")")

    #######################################################################
    #######################################################################
    #######################################################################
    
    #########################
    ### make gif for all bins
    ROOT.gSystem.Unlink("scan_pdfs.gif") ## remove old files
    ROOT.gSystem.Unlink("scan_cdfs.gif") ## remove old files
    ROOT.gSystem.Exec("/bin/rm -f scan_pdfs.gif scan_cdfs.gif") ## remove old files
    ROOT.gSystem.Exec("/bin/rm -rf /Users/noamtalhod/tmp/png") ## remove old files
    ROOT.gSystem.Exec("/bin/mkdir -p /Users/noamtalhod/tmp/png")
    
    ################################################
    ### initialize the shapes of all relevant slices
    print(f"\nBooking shapes...")
    NrawStepsIgnore = 10
    for ie in range(1,histos["SMALL_hdxinv_vs_E"].GetNbinsX()+1):
        for ixinv in range(1,histos["SMALL_hdxinv_vs_E"].GetNbinsY()+1):
            ### get the slice parameters
            label, E, x, NrawSteps = get_slice(ie,ixinv)
            ### skip if too few entries
            if(NrawSteps<NrawStepsIgnore): continue
            ### init the relevant model shapes
            shapes.update( {label : {"E":E, "x":x, "N":NrawSteps, "cnt_pdf":None, "sec_pdf":None, "cnt_cdf":None, "sec_cdf":None} } )
    

    #############################################
    ### collect the shapes of all relevant slices
    nCPUs = mp.cpu_count() if(parallelize) else 0
    print("nCPUs available:",nCPUs)
    ### Create a pool of workers
    pool = mp.Pool(nCPUs) if(parallelize) else None
    builds = {}
    for label,shape in shapes.items():
        E = shape["E"]
        X = shape["x"]
        N = shape["N"]
        P = par.GetModelPars(E,X)
        builds.update({label:P["build"]})
        print(f'Sending job: label={label}, build={P["build"]}, E={E*U.eV2MeV} MeV, X={X*U.cm2um} um, N={N} steps')
        ########################
        ### get the model shapes
        if(parallelize):
            pool.apply_async(add_slice_shapes, args=(E,X,P,N,label), callback=collect_shapes, error_callback=collect_errors)
        else:
            local_shapes = add_slice_shapes(E,X,P,N,label)
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
    print("\nPlotting shapes...")
    count = 0
    for ie in range(1,histos["SMALL_hdxinv_vs_E"].GetNbinsX()+1):
        for ixinv in range(1,histos["SMALL_hdxinv_vs_E"].GetNbinsY()+1):
            ### get the slice parameters
            label, E, x, NrawSteps = get_slice(ie,ixinv)
            ### skip if too few entries
            if(NrawSteps<NrawStepsIgnore): continue
            ### get the precalculated model shapes
            cnt_pdf = shapes[label]["cnt_pdf"]
            sec_pdf = shapes[label]["sec_pdf"]
            cnt_cdf = shapes[label]["cnt_cdf"]
            sec_cdf = shapes[label]["sec_cdf"]
            namecnt = "hdEcnt_"+label
            namesec = "hdEsec_"+label
            ##########################
            cgif_pdfs = ROOT.TCanvas("gif","",1000,1000)
            cgif_pdfs.Divide(2,2)
            cgif_pdfs.cd(1)
            ROOT.gPad.SetLogx()
            ROOT.gPad.SetLogy()
            ROOT.gPad.SetTicks(1,1)
            slices[namecnt].Draw("hist")
            if(cnt_pdf is not None):
                cnt_pdf_clone = cnt_pdf.Clone("cnt_model_clone")
                cnt_pdf_clone.Scale(slices[namecnt].GetMaximum() / cnt_pdf_clone.GetMaximum())
                cnt_pdf_clone.Draw("hist same")
            s = ROOT.TLatex() ### the text
            s.SetNDC(1);
            s.SetTextAlign(13);
            s.SetTextFont(22);
            s.SetTextColor(ROOT.kBlack)
            s.SetTextSize(0.04)
            modtitle = builds[label].replace("->"," #otimes ").replace(".","")
            modtitle = modtitle.replace(" #otimes SECB","")
            print(f"build={builds[label]}-->{modtitle}")
            s.DrawLatex(0.18,0.25,modtitle)
            ROOT.gPad.RedrawAxis()
            ##########################
            cgif_pdfs.cd(2)
            name = "hE_"+label
            ROOT.gPad.SetLogy()
            ROOT.gPad.SetTicks(1,1)
            slices[name].Draw("hist")
            s = ROOT.TLatex() ### the text
            s.SetNDC(1);
            s.SetTextAlign(13);
            s.SetTextFont(22);
            s.SetTextColor(ROOT.kBlack)
            s.SetTextSize(0.04)
            s.DrawLatex(0.15,0.86,ROOT.Form("E=%.3e #in [%.3e, %.3e) [MeV]" % (E*U.eV2MeV,slices["hE_"+label].GetXaxis().GetXmin(), slices["hE_"+label].GetXaxis().GetXmax())))
            s.DrawLatex(0.15,0.81,ROOT.Form("N raw steps = %d" % (NrawSteps)))
            ROOT.gPad.RedrawAxis()
            ROOT.gPad.Update()
            ##########################
            cgif_pdfs.cd(3)
            ROOT.gPad.SetLogx()
            ROOT.gPad.SetLogy()
            ROOT.gPad.SetTicks(1,1)
            slices[namesec].Draw("hist")
            if(sec_pdf is not None):
                sec_pdf_clone = sec_pdf.Clone("sec_model_clone")
                sec_pdf_clone.Scale(slices[namesec].GetMaximum() / sec_pdf_clone.GetMaximum())
                sec_pdf_clone.Draw("hist same")
                s = ROOT.TLatex() ### the text
                s.SetNDC(1);
                s.SetTextAlign(13);
                s.SetTextFont(22);
                s.SetTextColor(ROOT.kBlack)
                s.SetTextSize(0.04)
                modtitle = "SECB"
                s.DrawLatex(0.18,0.25,modtitle)
            ROOT.gPad.RedrawAxis()
            ROOT.gPad.Update()
            ##########################
            cgif_pdfs.cd(4)
            name = "hdx_"+label
            ROOT.gPad.SetLogy()
            ROOT.gPad.SetTicks(1,1)
            slices[name].Draw("hist")
            s = ROOT.TLatex() ### the text
            s.SetNDC(1);
            s.SetTextAlign(13);
            s.SetTextFont(22);
            s.SetTextColor(ROOT.kBlack)
            s.SetTextSize(0.04)
            s.DrawLatex(0.15,0.86,ROOT.Form("#Deltax=%.3e #in [%.3e, %.3e) [#mum]" % (x*U.cm2um,slices["hdx_"+label].GetXaxis().GetXmin(), slices["hdx_"+label].GetXaxis().GetXmax())))
            s.DrawLatex(0.15,0.81,ROOT.Form("N raw steps = %d" % (NrawSteps)))
            ROOT.gPad.RedrawAxis()
            ROOT.gPad.Update()
            ##########################
            cgif_pdfs.Update()
            cgif_pdfs.Print(f"/Users/noamtalhod/tmp/png/scan_pdfs_{count}.png")
            
            ##########################
            cgif_cdfs = ROOT.TCanvas("gif","",1000,1000)
            cgif_cdfs.Divide(2,2)
            cgif_cdfs.cd(1)
            ROOT.gPad.SetLogx()
            ROOT.gPad.SetLogy()
            ROOT.gPad.SetTicks(1,1)
            cnt_slice = slices[namecnt].Clone("cnt_slice_clone")
            if(cnt_slice.Integral()>0):
                cnt_slice.Scale(1./cnt_slice.Integral())
                cnt_slice.GetCumulative().Draw("hist")
            if(cnt_cdf is not None): cnt_cdf.Draw("hist same")
            s = ROOT.TLatex() ### the text
            s.SetNDC(1);
            s.SetTextAlign(13);
            s.SetTextFont(22);
            s.SetTextColor(ROOT.kBlack)
            s.SetTextSize(0.04)
            modtitle = builds[label].replace("->"," #otimes ").replace(".","")
            modtitle = modtitle.replace(" #otimes SECB","")
            print(f"build={builds[label]}-->{modtitle}")
            s.DrawLatex(0.18,0.25,modtitle)
            ROOT.gPad.RedrawAxis()
            ##########################
            cgif_cdfs.cd(2)
            name = "hE_"+label
            ROOT.gPad.SetLogy()
            ROOT.gPad.SetTicks(1,1)
            slices[name].Draw("hist")
            s = ROOT.TLatex() ### the text
            s.SetNDC(1);
            s.SetTextAlign(13);
            s.SetTextFont(22);
            s.SetTextColor(ROOT.kBlack)
            s.SetTextSize(0.04)
            s.DrawLatex(0.15,0.86,ROOT.Form("E=%.3e #in [%.3e, %.3e) [MeV]" % (E*U.eV2MeV,slices["hE_"+label].GetXaxis().GetXmin(), slices["hE_"+label].GetXaxis().GetXmax())))
            s.DrawLatex(0.15,0.81,ROOT.Form("N raw steps = %d" % (NrawSteps)))
            ROOT.gPad.RedrawAxis()
            ROOT.gPad.Update()
            ##########################
            cgif_cdfs.cd(3)
            ROOT.gPad.SetLogx()
            ROOT.gPad.SetLogy()
            ROOT.gPad.SetTicks(1,1)
            sec_slice = slices[namesec].Clone("sec_slice_clone")
            if(sec_slice.Integral()>0):
                sec_slice.Scale(1./sec_slice.Integral())
                sec_slice.GetCumulative().Draw("hist")
                s = ROOT.TLatex() ### the text
                s.SetNDC(1);
                s.SetTextAlign(13);
                s.SetTextFont(22);
                s.SetTextColor(ROOT.kBlack)
                s.SetTextSize(0.04)
                modtitle = "SECB"
                s.DrawLatex(0.18,0.25,modtitle)
            if(sec_cdf is not None): sec_cdf.Draw("hist same")
            ROOT.gPad.RedrawAxis()
            ROOT.gPad.Update()
            ##########################
            cgif_cdfs.cd(4)
            name = "hdx_"+label
            ROOT.gPad.SetLogy()
            ROOT.gPad.SetTicks(1,1)
            slices[name].Draw("hist")
            s = ROOT.TLatex() ### the text
            s.SetNDC(1);
            s.SetTextAlign(13);
            s.SetTextFont(22);
            s.SetTextColor(ROOT.kBlack)
            s.SetTextSize(0.04)
            s.DrawLatex(0.15,0.86,ROOT.Form("#Deltax=%.3e #in [%.3e, %.3e) [#mum]" % (x*U.cm2um,slices["hdx_"+label].GetXaxis().GetXmin(), slices["hdx_"+label].GetXaxis().GetXmax())))
            s.DrawLatex(0.15,0.81,ROOT.Form("N raw steps = %d" % (NrawSteps)))
            ROOT.gPad.RedrawAxis()
            ROOT.gPad.Update()
            ##########################
            cgif_cdfs.Update()
            cgif_cdfs.Print(f"/Users/noamtalhod/tmp/png/scan_cdfs_{count}.png")
            count += 1
    
    #####################
    ### finalize the gifs
    print("Making gif...")
    ROOT.gSystem.Exec("convert -delay 20 $(ls /Users/noamtalhod/tmp/png/scan_pdfs_*.png | sort -V) scan_pdfs.gif")
    ROOT.gSystem.Exec("convert -delay 20 $(ls /Users/noamtalhod/tmp/png/scan_cdfs_*.png | sort -V) scan_cdfs.gif")

    ###################################
    ### write everything to a root file
    print("Writing root file...")
    tfout = ROOT.TFile("scan.root","RECREATE")
    tfout.cd()
    for name,h in histos.items(): h.Write()
    for name,h in slices.items(): h.Write()
    tfout.Write()
    tfout.Close()
