import math
import array
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, rfft, irfft
from scipy.special import sici, exp1
from scipy.signal import convolve, fftconvolve
import ROOT

import constants as C
import units as U
import material as mat
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
parser.add_argument('-X', metavar='step size in x [um]', required=True,  help='step size in x [um]')
parser.add_argument('-W', metavar='fractional size in of the window around X:E', required=False,  help='fractional size of the window around X:E')
parser.add_argument('-N', metavar='N steps to process', required=False,  help='N steps to process')
argus = parser.parse_args()
EE = float(argus.E)
XX = float(argus.X)
WW = 0.01 if(argus.W is None) else float(argus.W)
NN = 0 if(argus.N is None) else int(argus.N)
print(f"Model with energy: {EE} [MeV], dx: {XX} [um], window: {WW*100} [%]")



ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.15)

ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning

# ROOT.Math.IntegratorOneDimOptions.SetDefaultIntegrator("GAUSS")
# ROOT.Math.IntegratorOneDimOptions.SetDefaultIntegrator("GAUSSLEGENDRE")
# ROOT.Math.IntegratorOneDimOptions.SetDefaultIntegrator("NONADAPTIVE")

#################################################
#################################################
#################################################
### GENERAL MODEL
Mat = mat.Si # or e.g. mat.Al
dEdxModel = "G4:Tcut" # or "BB:Tcut"
par = flct.Parameters(Mat.name+" parameters",C.mp,+1,Mat,dEdxModel,"inputs/eloss_p_si.txt","inputs/BB.csv")
modelpars = par.GetModelPars(EE*U.MeV2eV,XX*U.um2cm)
print(modelpars)

######################################################
######################################################
######################################################
### Build the model shapes
DOTIME = True
Mod = model.Model(XX*U.um2cm, EE*U.MeV2eV, modelpars, DOTIME)
Mod.set_fft_sampling_pars(N_t_bins=10000000,frac=0.05)
Mod.set_all_shapes()
cnt_pdfs = Mod.cnt_pdfs ## dict name-->TH1D
cnt_cdfs = Mod.cnt_cdfs ## dict name-->TH1D
sec_pdfs = Mod.sec_pdfs ## dict name-->TH1D
sec_cdfs = Mod.sec_cdfs ## dict name-->TH1D
# cnt_pdfs_scaled   = Mod.cnt_pdfs_scaled ## dict name-->TH1D
# cnt_cdfs_scaled   = Mod.cnt_cdfs_scaled ## dict name-->TH1D
# cnt_pdfs_scaled_arrx  = Mod.cnt_pdfs_scaled_arrx  ## np.array
# cnt_pdfs_scaled_arrsy = Mod.cnt_pdfs_scaled_arrsy ## dict name-->np.array
# cnt_cdfs_scaled_arrx  = Mod.cnt_cdfs_scaled_arrx  ## np.array
# cnt_cdfs_scaled_arrsy = Mod.cnt_cdfs_scaled_arrsy ## dict name-->np.array
# sec_pdfs_arrx  = Mod.sec_pdfs_arrx  ## np.array
# sec_pdfs_arrsy = Mod.sec_pdfs_arrsy ## dict name-->np.array
# sec_cdfs_arrx  = Mod.sec_cdfs_arrx  ## np.array
# sec_cdfs_arrsy = Mod.sec_cdfs_arrsy ## dict name-->np.array

# cnt_pdfs = Mod.get_model_pdfs()
# cnt_cdfs = Mod.get_cdfs(cnt_pdfs)
# sec_pdfs = Mod.get_secondaries_pdfs()
# sec_cdfs = Mod.get_cdfs(sec_pdfs)

######################################################
######################################################
######################################################
### Generate the toy data
ToyMC = toymc.ToyMC(XX*U.um2cm, EE*U.MeV2eV, Mod)
histos = ToyMC.Generate(Nsteps=1000000)

######################################################
######################################################
######################################################

### plot psi(t)
if(Mod.BEBL):
    canvas = ROOT.TCanvas("canvas", "canvas", 500,500)
    canvas.SaveAs("toymc_example.pdf(")
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
    canvas.SaveAs("toymc_example.pdf(")



canvas = ROOT.TCanvas("canvas", "canvas", 1400,1000)
canvas.Divide(3,2)
canvas.cd(1)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogy()
if(Mod.doLogx): ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.IONB and not Mod.BEBL):
    histos["hIon_non_gaus"].DrawNormalized("ep")
    cnt_pdfs["hBorysov_Ion"].DrawNormalized("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(2)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogy()
if(Mod.doLogx): ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.EX1B and not Mod.BEBL):
    histos["hExc_non_gaus"].DrawNormalized("ep")
    cnt_pdfs["hBorysov_Exc"].DrawNormalized("hist same")
canvas.cd(3)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogy()
if(Mod.doLogx): ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.SECB and not Mod.BEBL):
    histos["hSecondaries"].DrawNormalized("ep")
    sec_pdfs["hBorysov_Sec"].DrawNormalized("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(4)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogy()
if(Mod.doLogx): ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.IONG and not Mod.BEBL): 
    histos["hIon_gaus"].DrawNormalized("ep")
    cnt_pdfs["hTrncGaus_Ion"].DrawNormalized("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(5)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogy()
if(Mod.doLogx): ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.EX1G and not Mod.BEBL): 
    histos["hExc_gaus"].DrawNormalized("ep")
    cnt_pdfs["hTrncGaus_Exc"].DrawNormalized("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(6)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogy()
if(Mod.doLogx): ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
histos["hTotal"].DrawNormalized("ep")
cnt_pdfs["hModel"].DrawNormalized("hist same")
ROOT.gPad.RedrawAxis()
canvas.SaveAs("toymc_example.pdf")


canvas = ROOT.TCanvas("canvas", "canvas", 1400,1000)
canvas.Divide(3,2)
canvas.cd(1)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogy()
if(Mod.doLogx): ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.IONB and not Mod.BEBL):
    histos["hIon_non_gaus"].Scale(1./histos["hIon_non_gaus"].Integral())
    histos["hIon_non_gaus"].GetCumulative().Draw("ep")
    cnt_cdfs["hBorysov_Ion"].Draw("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(2)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogy()
if(Mod.doLogx): ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.EX1B and not Mod.BEBL):
    histos["hExc_non_gaus"].Scale(1./histos["hExc_non_gaus"].Integral())
    histos["hExc_non_gaus"].GetCumulative().Draw("ep")
    cnt_cdfs["hBorysov_Exc"].Draw("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(3)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogy()
if(Mod.doLogx): ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.SECB and not Mod.BEBL):
    histos["hSecondaries"].Scale(1./histos["hSecondaries"].Integral())
    histos["hSecondaries"].GetCumulative().Draw("ep")
    sec_cdfs["hBorysov_Sec"].Draw("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(4)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogy()
if(Mod.doLogx): ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.IONG and not Mod.BEBL): 
    histos["hIon_gaus"].Scale(1./histos["hIon_gaus"].Integral())
    histos["hIon_gaus"].GetCumulative().Draw("ep")
    cnt_cdfs["hTrncGaus_Ion"].Draw("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(5)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogy()
if(Mod.doLogx): ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.EX1G and not Mod.BEBL):
    histos["hExc_gaus"].Scale(1./histos["hExc_gaus"].Integral())
    histos["hExc_gaus"].GetCumulative().Draw("ep")
    cnt_cdfs["hTrncGaus_Exc"].Draw("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(6)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogy()
if(Mod.doLogx): ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
histos["hTotal"].Scale(1./histos["hTotal"].Integral())
histos["hTotal"].GetCumulative().Draw("ep")
cnt_cdfs["hModel"].Draw("hist same")
ROOT.gPad.RedrawAxis()
canvas.SaveAs("toymc_example.pdf)")


print(f"Done")