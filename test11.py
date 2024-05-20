import math
import array
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, rfft, irfft
from scipy.special import sici, exp1
from scipy.signal import convolve, fftconvolve
import ROOT
import units as U
import bins
import model
import toymc

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

# slicename = "E20.0MeV_X0.02um"
# slicename = "E90.0MeV_X5.0um"
# slicename = "E50.0MeV_X2.0um"
# slicename = "E3.0MeV_X0.0004um"
slicename = "E10.0MeV_X1.0um"
# slicename = "E98.0MeV_X0.1um"
# slicename = "E70.0MeV_X40.0um"
# slicename = "E30.0MeV_X20.0um"
# slicename = "E75.0MeV_X0.5um"

slices = {
"E90.0MeV_X5.0um"  :{"build":"ION.B->EX1.B->ION.G", "param":{"minLoss":10, "meanLoss":4637.3011671010145, "w3":54.07302799365922,  "p3":6.587055235645483,  "w":0.9453807798043846, "n1":4.398932737311463, "e1":309.2284174928461, "ion_mean":635.8774864416207, "ion_sigma":58.63353021319207}},
"E10.0MeV_X1.0um"  :{"build":"ION.B->ION.G->EX1.G", "param":{"minLoss":10, "meanLoss":5432.4800000000005, "w3":61.23531984870318,  "p3":6.76186610770327,   "w":0.9381461415669665, "ex1_mean":1593.5274666666667, "ex1_sigma":728.7464244543313, "ion_mean":799.8150657098756, "ion_sigma":72.7383213329989}},
"E50.0MeV_X2.0um"  :{"build":"ION.B->EX1.B->ION.G", "param":{"minLoss":10, "meanLoss":2914.8354253007374, "w3":38.17328908112162,  "p3":5.964541658914232,  "w":0.9614411221402812, "n1":3.4373073160268537, "e1":248.74656607947875, "ion_mean":317.2292300230526, "ion_sigma":30.23505532400481}},
"E1.0MeV_X0.2um"   :{"build":"ION.B->ION.G->EX1.G", "param":{"minLoss":10, "meanLoss":6850.1400000000000, "w3":73.73612227172656,  "p3":6.985611814959718,  "w":0.9255190684123974, "ex1_mean":2009.3744, "ex1_sigma":864.698998952752, "ion_mean":1111.923209679792, "ion_sigma":98.84963373342933}},
"E20.0MeV_X0.02um" :{"build":"ION.B->EX1.B",        "param":{"minLoss":10, "meanLoss":30370424.818535507, "w3":10.0,               "p3":0.4885079866018566, "w":0.98989898989899,   "n1":0.3513529553925317, "e1":50.71059064704698}},
"E30.0MeV_X20.0um" :{"build":"ION.B->ION.G->EX1.G", "param":{"minLoss":10, "meanLoss":44253.118737987970, "w3":313.84677497953857, "p3":7.82413016148422,   "w":0.6829830555762236, "ex1_mean":12980.914829809804, "ex1_sigma":2997.130804991398, "ion_mean":12390.801010012685, "ion_sigma":807.0696909976394}},
"E2.0MeV_X0.01um"  :{"build":"ION.B->EX1.B",        "param":{"minLoss":10, "meanLoss":198.28645967133156, "w3":10.0,               "p3":1.5947178836519698, "w":0.98989898989899,   "n1":0.748902373766523, "e1":77.66570144213537}},
"E98.0MeV_X0.1um"  :{"build":"ION.B->EX1.B",        "param":{"minLoss":10, "meanLoss":86.40112466840405,  "w3":10.0,               "p3":0.6948806232393914, "w":0.98989898989899,   "n1":0.44348807313992833, "e1":57.14771475880315}},
"E3.0MeV_X0.0004um":{"build":"BEBL",                "param":{"minLoss":10, "meanLoss":5.8000952819077790, "w3":10.0,               "p3":0.04664723798223652,"w":0.98989898989899,  "n1":0.061589263836202356, "e1":27.624315939507678}},
"E70.0MeV_X40.0um" :{"build":"ION.B->ION.G->EX1.G", "param":{"minLoss":10, "meanLoss":45058.795925837510, "w3":317.64207634634414, "p3":7.827206893539028,  "w":0.6791494178319757, "ex1_mean":13217.246804912334, "ex1_sigma":3024.290791078023, "ion_mean":12660.39362713598, "ion_sigma":821.0685861711323}},
"E75.0MeV_X0.5um"  :{"build":"ION.B->EX1.B",        "param":{"minLoss":10, "meanLoss":532.70321455460260, "w3":10.0,               "p3":4.284263001806674,  "w":0.98989898989899,    "n1":1.344246186425751, "e1":116.24329767909988}},
}

param = slices[slicename]["param"]
build = slices[slicename]["build"]

E         = float( slicename.split("_")[0].replace("E","").replace("MeV","") )*U.MeV2eV
dx_cm     = float( slicename.split("_")[1].replace("X","").replace("um","") )*U.um2cm
minLoss   = param["minLoss"]     if("minLoss"   in param) else -1
meanLoss  = param["meanLoss"]    if("meanLoss"  in param) else -1
w3        = param["w3"]          if("w3"        in param) else -1
p3        = param["p3"]          if("p3"        in param) else -1
w         = param["w"]           if("w"         in param) else -1
e1        = param["e1"]          if("e1"        in param) else -1
n1        = param["n1"]          if("n1"        in param) else -1
ion_mean  = param["ion_mean"]    if("ion_mean"  in param) else -1
ion_sigma = param["ion_sigma"]   if("ion_sigma" in param) else -1
ex1_mean  = param["ex1_mean"]    if("ex1_mean"  in param) else -1
ex1_sigma = param["ex1_sigma"]   if("ex1_sigma" in param) else -1


######################################################
######################################################
######################################################
Mod = model.Model(build,dx_cm,E,minLoss,meanLoss,w3,p3,w,e1,n1,ex1_mean,ex1_sigma,ion_mean,ion_sigma)
pdfs = Mod.get_model_pdfs()

######################################################
######################################################
######################################################
ToyMC = toymc.ToyMC(E,dx_cm,Mod)
histos = ToyMC.Generate(Nsteps=1000000)
    

######################################################
######################################################
######################################################

### plot psi(t)
if(Mod.BEBL):
    canvas = ROOT.TCanvas("canvas", "canvas", 500,500)
    canvas.SaveAs("test11.pdf(")
else:
    canvas = ROOT.TCanvas("canvas", "canvas", 1000,500)
    canvas.Divide(2,1)
    canvas.cd(1)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridx()
    ROOT.gPad.SetGridy()
    ROOT.gPad.SetLeftMargin(0.15)
    ROOT.gPad.SetRightMargin(0.1)
    Mod.psiRe.Draw("hist")
    ROOT.gPad.RedrawAxis()
    canvas.cd(2)
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetGridx()
    ROOT.gPad.SetGridy()
    ROOT.gPad.SetLeftMargin(0.15)
    ROOT.gPad.SetRightMargin(0.1)
    Mod.psiIm.Draw("hist")
    ROOT.gPad.RedrawAxis()
    canvas.SaveAs("test11.pdf(")



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
if(Mod.IONB):
    histos["hIon_non_gaus"].DrawNormalized("ep")
    pdfs["hBorysov_Ion"].DrawNormalized("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(2)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogy()
if(Mod.doLogx): ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.EX1B):
    histos["hExc_non_gaus"].DrawNormalized("ep")
    pdfs["hBorysov_Exc"].DrawNormalized("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(4)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogy()
if(Mod.doLogx): ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.IONG): 
    histos["hIon_gaus"].DrawNormalized("ep")
    pdfs["hTrncGaus_Ion"].DrawNormalized("hist same")
ROOT.gPad.RedrawAxis()
canvas.cd(5)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogy()
if(Mod.doLogx): ROOT.gPad.SetLogx()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.1)
if(Mod.EX1G): 
    histos["hExc_gaus"].DrawNormalized("ep")
    pdfs["hTrncGaus_Exc"].DrawNormalized("hist same")
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
pdfs["hModel"].DrawNormalized("hist same")
ROOT.gPad.RedrawAxis()
canvas.SaveAs("test11.pdf)")


print(f"Done")