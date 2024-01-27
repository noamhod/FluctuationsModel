import math
import ROOT
from ROOT import TH1D, TCanvas, TRandom
import constants as C
import units as U
import material as mat
import fluctuations as flct
import shapes

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.15)

ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning

rho_Si = 2.329     # Silicon, g/cm3
Z_Si   = [14]      # Silicon atomic number (Z)
A_Si   = [28.0855] # Silicon atomic mass (A)
I_Si   = 173.0     # Silicon mean excitation energy (I), eV
Ep_Si  = 31.05     # Silicon plasma energy (E_p), eV
Tc_Si  = 990       # Silicon, production threshold for delta ray, eV
den_Si = [31.055, 2.103, 4.4351, 0.2014, 2.8715, 0.14921, 3.2546, 0.14, 0.059, 173.]
nel_Si = 1
Si = mat.Material("Silicon","Si",rho_Si,Z_Si,A_Si,I_Si,Tc_Si,den_Si,nel_Si)
dEdxModel = "BB:Tcut"
par = flct.Parameters("Silicon parameters",C.mp,+1,Si,dEdxModel,"eloss_p_si.txt","BB.csv")
# func = shapes.Functions("Landau")

### scenario
dEmin  = 1e-4
dEmax  = 1e-2
dEbins = 5000
E = 90*U.MeV2eV
X = 2*U.um2cm

### model
model = par.DifferentialModel(E,X,doSec=False)
n1 = par.n12_mean(E,X,1)
n2 = par.n12_mean(E,X,2)
n3 = par.n3_mean(E,X)
meanG  = model["Gauss"]["mean"]*U.eV2MeV
widthG = model["Gauss"]["width"]*U.eV2MeV
mpvL   = model["Landau"]["mpv"]*U.eV2MeV
widthL = model["Landau"]["width"]*U.eV2MeV
print(f"n1={n1}, n2={n2}, n3={n3}")
print(model)

### extended model
w1 = par.mat.Tc/par.E0
alpha  = (w1*(par.ncontmax+n3))/(w1*par.ncontmax+n3)
alpha1 = alpha*math.log(alpha)/(alpha-1)
naAvg = n3*w1*(alpha-1)/(alpha*(w1-1))
dEAvg = (model["Gauss"]["mean"] + naAvg*par.E0*alpha1)
sigdE = math.sqrt(model["Gauss"]["width"]**2 + naAvg*(alpha-alpha1**2)*(par.E0**2))
p3 = n3 - naAvg
w3 = alpha*par.E0
w = (par.mat.Tc-w3)/par.mat.Tc if(par.mat.Tc>w3) else 0
print(f"w1={w1}")
print(f"alpha={alpha}")
print(f"alpha1={alpha1}")
print(f"naAvg={naAvg}")
print(f"dEAvg={dEAvg}")
print(f"sigdE={sigdE}")
print(f"p3={p3}")
print(f"w3={w3}")
meanG  = dEAvg*U.eV2MeV
widthG = sigdE*U.eV2MeV


### generate the data
rnd = TRandom()
rnd.SetSeed()
Nparticles = 1000000
hdE = TH1D("hdE","",dEbins,dEmin,dEmax)
for i in range(Nparticles):
    ### excitation can be either Poisson or Gaussain
    N1 = rnd.Poisson(n1)
    E1 = par.E1*U.eV2MeV
    # E1 = rnd.Landau(model["Landau"]["mpv"]*U.eV2MeV,model["Landau"]["width"]*U.eV2MeV)
    
    ### ionization is ~always Poisson and sometimes Gaussian
    ### First the Poisson part
    E3L = 0
    if(par.mat.Tc>w3):
        N3B = rnd.Poisson(p3)
        if(N3B>0):
            for j in range(N3B):
                u = rnd.Uniform()
                E3L += w3/(1-w*u)
    E3L *= U.eV2MeV
    ### Then the Gaussian part
    E3G = rnd.Gaus(meanG,widthG)
    while(E3G<0 or E3G>2*meanG):
        E3G = rnd.Gaus(meanG,widthG)
    ### And finally combine the two parts
    E3 = E3L + E3G
    
    ### The total loss is the sum of excitation and ionization
    ELOSS = N1*E1 + E3
    hdE.Fill(ELOSS)

### RooFit basic variable
dE = ROOT.RooRealVar("dE", "#DeltaE [MeV]", dEmin, dEmax)
dE.setBins(dEbins, "cache") ## set #bins to be used for FFT sampling to 10000
### RooFit shapes
landau_mpv = ROOT.RooRealVar("landau_mpv", "mean landau", mpvL)
landau_sigma = ROOT.RooRealVar("landau_sigma", "sigma landau", widthL)
landau_model = ROOT.RooLandau("landau_model", "lx", dE, landau_mpv, landau_sigma)
gauss_mean = ROOT.RooRealVar("gauss_mean", "mg", meanG)
gauss_sigma = ROOT.RooRealVar("gauss_sigma", "sg", widthG)
gauss_model = ROOT.RooGaussian("gauss_model", "gauss", dE, gauss_mean, gauss_sigma)
langaus_convolution_model = ROOT.RooFFTConvPdf("landauXgauss_model", "landau(x)gauss", dE, landau_model, gauss_model)
### RooFit data
data = ROOT.RooDataHist("data","dataset",dE,hdE)
### RooFit plot
frame = dE.frame(ROOT.RooFit.Title("landau #otimes gauss convolution"))
data.plotOn(frame, ROOT.RooFit.Binning(dEbins), ROOT.RooFit.Name("data"))
landau_model.plotOn(frame, ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kGreen), ROOT.RooFit.Name("landau_model"))
gauss_model.plotOn(frame, ROOT.RooFit.LineStyle(ROOT.kDashed), ROOT.RooFit.LineColor(ROOT.kRed), ROOT.RooFit.Name("gauss_model"))
langaus_convolution_model.plotOn(frame, ROOT.RooFit.LineColor(ROOT.kBlue), ROOT.RooFit.Name("landauXgauss_model"))

### final plot
canvas = ROOT.TCanvas("canvas", "canvas", 100, 100, 800, 600)
legend = ROOT.TLegend(0.2, 0.6, 0.4, 0.85)
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
frame.GetYaxis().SetLabelOffset(0.008)
frame.GetYaxis().SetTitleOffset(1.5)
frame.GetYaxis().SetTitleSize(0.045)
frame.GetXaxis().SetTitleSize(0.045)
frame.Draw()
legend.AddEntry("data", "data", 'LEP')
legend.AddEntry("landauXgauss_model", "Convolution", 'L')
legend.AddEntry("landau_model", "Landau", 'L')
legend.AddEntry("gauss_model", "Gauss", 'L')
legend.Draw()
canvas.SaveAs("testconv.pdf")