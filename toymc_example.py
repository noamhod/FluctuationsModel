import math
import array
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, rfft, irfft
from scipy.special import sici, exp1
from scipy.signal import convolve, fftconvolve
import ROOT
import pickle

import constants as C
import units as U
import material as mat
import particle as prt
import bins
import fluctuations as flct
import hist
import model
import toymc

#################################################
#################################################
#################################################
import argparse
parser = argparse.ArgumentParser(description='toymc_example.py...')
parser.add_argument('-E', metavar='incoming particle energy [MeV]', required=True,  help='incoming particle energy [MeV]')
parser.add_argument('-L', metavar='step size in L [um]', required=True,  help='step size in L [um]')
parser.add_argument('-N', metavar='N steps to process', required=False,  help='N steps to process')
argus = parser.parse_args()
EE = float(argus.E)
LL = float(argus.L)
NN = 0 if(argus.N is None) else int(argus.N)
print(f"Model with energy: {EE} [MeV] and dL: {LL} [um]")



ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.15)

ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning


#################################################
#################################################
#################################################
### GENERAL MODEL
dEdxModel  = "G4:Tcut" # or "BB:Tcut"
TargetMat  = mat.Si # or e.g. mat.Al
PrimaryPrt = prt.Particle(name="proton",meV=938.27208816*U.MeV2eV,mamu=1.007276466621,chrg=+1.,spin=0.5,lepn=0,magm=2.79284734463)
par        = flct.Parameters(PrimaryPrt,TargetMat,dEdxModel,"inputs/dEdx_p_si.txt")
modelpars  = par.GetModelPars(EE*U.MeV2eV,LL*U.um2cm)
print(modelpars)

######################################################
######################################################
######################################################
### Build the model shapes
DOTIME = True
Mod = model.Model(LL*U.um2cm, EE*U.MeV2eV, modelpars, DOTIME)
# Mod.set_fft_sampling_pars(N_t_bins=10000000,frac=0.05)
Mod.set_fft_sampling_pars_rotem(N_t_bins=10000000,frac=0.05)
# Mod.set_all_shapes()
Mod.set_continuous_shapes()
Mod.set_secondaries_shapes()
cnt_pdfs = Mod.cnt_pdfs ## dict name-->TH1D
cnt_cdfs = Mod.cnt_cdfs ## dict name-->TH1D
sec_pdfs = Mod.sec_pdfs ## dict name-->TH1D
sec_cdfs = Mod.sec_cdfs ## dict name-->TH1D


######################################################
######################################################
######################################################
### Generate the toy data
ToyMC = toymc.ToyMC(LL*U.um2cm, EE*U.MeV2eV, Mod)
histos = ToyMC.Generate(Nsteps=1000000)

######################################################
######################################################
######################################################

pdffilename = "toymc_example.pdf"

pklfilename = "toymc_example.pkl"
fpkl = open(pklfilename,"wb")
shapes = {}
shapes.update({"x":Mod.cnt_pdfs_arrx})
shapes.update({"y":Mod.cnt_pdfs_arrsy})
shapes.update({"x_scl":Mod.cnt_pdfs_scaled_arrx})
shapes.update({"y_slc":Mod.cnt_pdfs_scaled_arrsy})
pickle.dump(shapes, fpkl, protocol=pickle.HIGHEST_PROTOCOL) ### dump to pickle
fpkl.close()



### plot psi(t)
if(Mod.BEBL or Mod.TGAU or Mod.TGAM):
    canvas = ROOT.TCanvas("canvas", "canvas", 500,500)
    canvas.SaveAs(pdffilename+"(")
else:
    if(Mod.psiRe is None or Mod.psiIm is None):
        trange,psiRe,psiIm = Mod.scipy_psi_of_t("psi_of_t")
    hpsiRe,hpsiIm = Mod.scipy_psi_of_t_as_h("psi_of_t")
    canvas = ROOT.TCanvas("canvas", "canvas", 1000,500)
    canvas.Divide(2,1)
    canvas.cd(1)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridx()
    ROOT.gPad.SetGridy()
    ROOT.gPad.SetLeftMargin(0.15)
    ROOT.gPad.SetRightMargin(0.1)
    hpsiRe.Draw("hist")
    ROOT.gPad.RedrawAxis()
    canvas.cd(2)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridx()
    ROOT.gPad.SetGridy()
    ROOT.gPad.SetLeftMargin(0.15)
    ROOT.gPad.SetRightMargin(0.1)
    hpsiIm.Draw("hist")
    ROOT.gPad.RedrawAxis()
    canvas.SaveAs(pdffilename+"(")



canvas = ROOT.TCanvas("canvas", "canvas", 1500,1000)
canvas.Divide(3,2)
canvas.cd(1)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.IONB and not (Mod.BEBL or Mod.TGAU or Mod.TGAM)):
    if(histos["hIon_non_gaus"].Integral()>0): ROOT.gPad.SetLogy()
    # histos["hIon_non_gaus"].DrawNormalized("ep")
    # cnt_pdfs["hBorysov_Ion"].DrawNormalized("hist same")
    hist.reset_hrange_left(histos["hIon_non_gaus"],1e-2).DrawNormalized("ep")
    hist.reset_hrange_left(cnt_pdfs["hBorysov_Ion"],1e-2).DrawNormalized("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(2)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.EX1B and not (Mod.BEBL or Mod.TGAU or Mod.TGAM)):
    if(histos["hExc_non_gaus"].Integral()>0): ROOT.gPad.SetLogy()
    # histos["hExc_non_gaus"].DrawNormalized("ep")
    # cnt_pdfs["hBorysov_Exc"].DrawNormalized("hist same")
    hist.reset_hrange_left(histos["hExc_non_gaus"],1e-2).DrawNormalized("ep")
    hist.reset_hrange_left(cnt_pdfs["hBorysov_Exc"],1e-2).DrawNormalized("hist same")
canvas.cd(3)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.SECB):# and not (Mod.BEBL or Mod.TGAU or Mod.TGAM)):
    if(histos["hSecondaries"].Integral()>0): ROOT.gPad.SetLogy()
    # histos["hSecondaries"].DrawNormalized("ep")
    # sec_pdfs["hBorysov_Sec"].DrawNormalized("hist same")
    hist.reset_hrange_left(histos["hSecondaries"],1e-2).DrawNormalized("ep")
    hist.reset_hrange_left(sec_pdfs["hBorysov_Sec"],1e-2).DrawNormalized("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(4)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.IONG and not (Mod.BEBL or Mod.TGAU or Mod.TGAM)): 
    if(histos["hIon_gaus"].Integral()>0): ROOT.gPad.SetLogy()
    # histos["hIon_gaus"].DrawNormalized("ep")
    # cnt_pdfs["hTrncGaus_Ion"].DrawNormalized("hist same")
    hist.reset_hrange_left(histos["hIon_gaus"],1e-2).DrawNormalized("ep")
    hist.reset_hrange_left(cnt_pdfs["hTrncGaus_Ion"],1e-2).DrawNormalized("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(5)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.EX1G and not (Mod.BEBL or Mod.TGAU or Mod.TGAM)):
    if(histos["hExc_gaus"].Integral()>0): ROOT.gPad.SetLogy()
    # histos["hExc_gaus"].DrawNormalized("ep")
    # cnt_pdfs["hTrncGaus_Exc"].DrawNormalized("hist same")
    hist.reset_hrange_left(histos["hExc_gaus"],1e-2).DrawNormalized("ep")
    hist.reset_hrange_left(cnt_pdfs["hTrncGaus_Exc"],1e-2).DrawNormalized("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(6)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
if(histos["hTotal"].Integral()>0): ROOT.gPad.SetLogy()
ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
# histos["hTotal"].DrawNormalized("ep")
# cnt_pdfs["hModel"].DrawNormalized("hist same")
hist.reset_hrange_left(histos["hTotal"],1e-2).DrawNormalized("ep")
hist.reset_hrange_left(cnt_pdfs["hModel"],1e-2).DrawNormalized("hist same")
ROOT.gPad.RedrawAxis()
canvas.SaveAs(pdffilename)




canvas = ROOT.TCanvas("canvas", "canvas", 1500,1000)
canvas.Divide(3,2)
canvas.cd(1)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.IONB and not (Mod.BEBL or Mod.TGAU or Mod.TGAM)):
    if(histos["hIon_non_gaus"].Integral()>0): ROOT.gPad.SetLogy()
    histos["hIon_non_gaus"].Scale(1./histos["hIon_non_gaus"].Integral())
    histos["hIon_non_gaus"].GetCumulative().Draw("ep")
    # hist.reset_hrange_left(histos["hIon_non_gaus"].GetCumulative(),1e-2).Draw("ep")
    cnt_cdfs["hBorysov_Ion"].Draw("hist same")
    # hist.reset_hrange_left(cnt_cdfs["hBorysov_Ion"],1e-2).Draw("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(2)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.EX1B and not (Mod.BEBL or Mod.TGAU or Mod.TGAM)):
    if(histos["hExc_non_gaus"].Integral()>0): ROOT.gPad.SetLogy()
    histos["hExc_non_gaus"].Scale(1./histos["hExc_non_gaus"].Integral())
    histos["hExc_non_gaus"].GetCumulative().Draw("ep")
    # hist.reset_hrange_left(histos["hExc_non_gaus"].GetCumulative(),1e-2).Draw("ep")
    cnt_cdfs["hBorysov_Exc"].Draw("hist same")
    # hist.reset_hrange_left(cnt_cdfs["hBorysov_Exc"],1e-2).Draw("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(3)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.SECB):# and not (Mod.BEBL or Mod.TGAU or Mod.TGAM)):
    if(histos["hSecondaries"].Integral()>0): ROOT.gPad.SetLogy()
    histos["hSecondaries"].Scale(1./histos["hSecondaries"].Integral())
    histos["hSecondaries"].GetCumulative().Draw("ep")
    # hist.reset_hrange_left(histos["hSecondaries"].GetCumulative(),1e-2).Draw("ep")
    sec_cdfs["hBorysov_Sec"].Draw("hist same")
    # hist.reset_hrange_left(sec_cdfs["hBorysov_Sec"],1e-2).Draw("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(4)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.IONG and not (Mod.BEBL or Mod.TGAU or Mod.TGAM)):
    if(histos["hIon_gaus"].Integral()>0): ROOT.gPad.SetLogy()
    histos["hIon_gaus"].Scale(1./histos["hIon_gaus"].Integral())
    histos["hIon_gaus"].GetCumulative().Draw("ep")
    # hist.reset_hrange_left(histos["hIon_gaus"].GetCumulative(),1e-2).Draw("ep")
    cnt_cdfs["hTrncGaus_Ion"].Draw("hist same")
    # hist.reset_hrange_left(cnt_cdfs["hTrncGaus_Ion"],1e-2).Draw("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(5)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.EX1G and not (Mod.BEBL or Mod.TGAU or Mod.TGAM)):
    if(histos["hExc_gaus"].Integral()>0): ROOT.gPad.SetLogy()
    histos["hExc_gaus"].Scale(1./histos["hExc_gaus"].Integral())
    histos["hExc_gaus"].GetCumulative().Draw("ep")
    # hist.reset_hrange_left(histos["hExc_gaus"].GetCumulative(),1e-2).Draw("ep")
    cnt_cdfs["hTrncGaus_Exc"].Draw("hist same")
    # hist.reset_hrange_left(cnt_cdfs["hTrncGaus_Exc"],1e-2).Draw("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(6)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(histos["hTotal"].Integral()>0): ROOT.gPad.SetLogy()
histos["hTotal"].Scale(1./histos["hTotal"].Integral())
histos["hTotal"].GetCumulative().Draw("ep")
# hist.reset_hrange_left(histos["hTotal"].GetCumulative(),1e-2).Draw("ep")
cnt_cdfs["hModel"].Draw("hist same")
# hist.reset_hrange_left(cnt_cdfs["hModel"],1e-2).Draw("hist same")
ROOT.gPad.RedrawAxis()
canvas.SaveAs(pdffilename+")")


### write to root file
fOut = ROOT.TFile(pdffilename.replace("pdf","root"), "RECREATE")
fOut.cd()
for name,h in histos.items():
    if(h is not None): h.Write()
for name,p in cnt_pdfs.items(): 
    if(p is not None): p.Write()
for name,p in sec_pdfs.items(): 
    if(p is not None): p.Write()
for name,c in cnt_cdfs.items(): 
    if(c is not None): c.Write()
for name,c in sec_cdfs.items(): 
    if(c is not None): c.Write()
fOut.Write()
fOut.Close()

print(f"Done")