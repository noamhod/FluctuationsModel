import ROOT
import numpy as np
import constants as C
import units as U
import material as mat
import bins
import fluctuations as flct
import shapes

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.15)

dEmin  = 1e-5#1e-4#7e-4
dEmax  = 2e-2#1e-2#7e-2
dEbins = 1000

###################################
###################################
###################################
### get the data
# dElst = [] ## for unbinned RooDataSet
hdE_fixed =  ROOT.TH1D("hdE_fixed","E=100 MeV, dx=5 #mum;#DeltaE [MeV];Particles", dEbins, dEmin, dEmax)
# with open("dEs.csv") as f:
# with open("dEs_100MeV_0p5um.csv") as f:
# with open("dEs_3MeV_0p5um.csv") as f:
with open("dEs_100MeV_1um.csv") as f:
# with open("dEs_100MeV_0p5um_10umCut.csv") as f:
    for line in f:
        if("#" in line): continue
        words = line.split(",")
        dE = float(words[1])*1e-6
        hdE_fixed.Fill(dE)
        # dElst.append(dE) ## for unbinned RooDataSet
# dEarr = np.array(dElst) ## for unbinned RooDataSet

###################################
###################################
###################################
### Model

### Silicon parameters: https://pdg.lbl.gov/2022/AtomicNuclearProperties/HTML/silicon_Si.html
rho_Si = 2.329     # Silicon, g/cm3
Z_Si   = [14]      # Silicon atomic number (Z)
A_Si   = [28.0855] # Silicon atomic mass (A)
I_Si   = 173.0     # Silicon mean excitation energy (I), eV
Ep_Si  = 31.05     # Silicon plasma energy (E_p), eV
Tc_Si  = 990       # Silicon, production threshold for delta ray, eV
den_Si = [31.055, 2.103, 4.4351, 0.2014, 2.8715, 0.14921, 3.2546, 0.14, 0.059, 173.]
nel_Si = 1
Si = mat.Material("Silicon","Si",rho_Si,Z_Si,A_Si,I_Si,Tc_Si,den_Si,nel_Si)

par = flct.Parameters("Silicon parameters",C.mp,+1,Si,"eloss_p_si.txt","BB.csv","low")

E = 100*U.MeV2eV #3*U.MeV2eV #100*U.MeV2eV
X = 1*U.um2cm #0.5*U.um2cm #5*U.um2cm
func = shapes.Functions("Landau")
Delta_p,Width,Model = par.Model(E,X)
Delta_p = Delta_p*U.eV2MeV
Width   = Width*U.eV2MeV
Gwidth  = par.WidthThick(E,X)*U.eV2MeV
Delta_p_PDG = par.Delta_p_PDG(E,X)*U.eV2MeV
print(f"Delta_p={Delta_p}, Delta_p_PDG={Delta_p_PDG}, Width={Width}, Gwidth={Gwidth}")
# function = func.fLandau(dEmin,dEmax,[Delta_p, Width, 1],"fixed")
# hLandau = func.f2h(function,hdE_fixed)
# hdEmaximum = hdE_fixed.GetBinContent( hdE_fixed.GetMaximumBin() )
# hLandaumaximum = hLandau.GetBinContent( hLandau.GetMaximumBin() )
# hLandau.Scale(hdEmaximum/hLandaumaximum)
# hLandau.SetFillColorAlpha(ROOT.kRed,0.2)


###################################
###################################
###################################
### RooFit
de = ROOT.RooRealVar("de", "#DeltaE [MeV]", dEmin, dEmax)
ml = ROOT.RooRealVar("ml", "mean landau", 4.3e-3, 1e-5, 1e-1)
sl = ROOT.RooRealVar("sl", "sigma landau", 1e-3, 1e-5, 1e-1)
landau = ROOT.RooLandau("lx", "lx", de, ml, sl)

mg = ROOT.RooRealVar("mg", "mg", 0)
sg = ROOT.RooRealVar("sg", "sg", 1e-3, 1e-5, 1e-1)
gauss = ROOT.RooGaussian("gauss", "gauss", de, mg, sg)

# Set #bins to be used for FFT sampling to 10000
de.setBins(dEbins, "cache")

# Construct landau (x) gauss
convolution = ROOT.RooFFTConvPdf("lxg", "landau (X) gauss", de, landau, gauss)

Gwidth  = Width
Delta_p *= 1.
Width   *= 1. #1./2.

# fixed PDFs from the model parameters
ml_model = ROOT.RooRealVar("ml_model", "mean landau", Delta_p)
sl_model = ROOT.RooRealVar("sl_model", "sigma landau", Width)
landau_model = ROOT.RooLandau("lx_model", "lx", de, ml_model, sl_model)
mg_model = ROOT.RooRealVar("mg_model", "mg", 0)
sg_model = ROOT.RooRealVar("sg_model", "sg", Gwidth)
gauss_model = ROOT.RooGaussian("gauss_model", "gauss", de, mg_model, sg_model)
convolution_model = ROOT.RooFFTConvPdf("lxg_model", "landau (X) gauss", de, landau_model, gauss_model)

# Import the data to a RooDataSet, passing a dictionary of arrays and the
# corresponding RooRealVars just like you would pass to the RooDataSet constructor.
# data = ROOT.RooDataSet.from_numpy({"de": dEarr}, [de])

data = ROOT.RooDataHist("data","dataset",de,hdE_fixed)

convolution.fitTo(data)

## RooPlot
frame1 = de.frame(ROOT.RooFit.Title("landau #otimes gauss convolution"))
data.plotOn(frame1, ROOT.RooFit.Binning(dEbins), ROOT.RooFit.Name("data"))
### fit
convolution.plotOn(frame1, ROOT.RooFit.Name("lxg"))
gauss.plotOn(frame1, ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.Name("gauss"))
landau.plotOn(frame1, ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Name("landau"))
### model
convolution_model.plotOn(frame1, ROOT.RooFit.LineColor(ROOT.kOrange), ROOT.RooFit.Name("lxg_model"))
landau_model.plotOn(frame1, ROOT.RooFit.LineStyle(ROOT.kDotted), ROOT.RooFit.LineColor(ROOT.kOrange), ROOT.RooFit.Name("gauss_model"))
gauss_model.plotOn(frame1, ROOT.RooFit.LineStyle(ROOT.kDashDotted), ROOT.RooFit.LineColor(ROOT.kOrange), ROOT.RooFit.Name("lx_model"))

canvas = ROOT.TCanvas("canvas", "canvas", 100, 100, 800, 600)
legend = ROOT.TLegend(0.7, 0.6, 0.85, 0.85)
legend.SetTextSize(0.032)
legend.SetBorderSize(0)
legend.SetFillStyle(0)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.02)
frame1.GetYaxis().SetLabelOffset(0.008)
frame1.GetYaxis().SetTitleOffset(1.5)
frame1.GetYaxis().SetTitleSize(0.045)
frame1.GetXaxis().SetTitleSize(0.045)
frame1.Draw()
legend.AddEntry("data", "data", 'LEP')
legend.AddEntry("lxg", "convolution", 'L')
legend.AddEntry("landau", "landau", 'L')
legend.AddEntry("gauss", "gauss", 'L')
legend.AddEntry("lxg_model", "convolution (model)", 'L')
legend.AddEntry("lx_model", "landau (model)", 'L')
legend.AddEntry("gauss_model", "gauss (model)", 'L')
legend.Draw()
# hLandau.Draw("hist same")
canvas.SaveAs("test_langaus.pdf")


print(f"par.isThick(E,X)={par.isThick(E,X)}")
print(f"par.Wmax(E)={par.Wmax(E)}")
print(f"par.Mean(E,X)={par.Mean(E,X)}")