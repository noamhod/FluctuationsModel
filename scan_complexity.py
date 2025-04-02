import math
import array
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, rfft, irfft
from scipy.special import sici, exp1
from scipy.signal import convolve, fftconvolve
from scipy.interpolate import interp1d
from scipy.fft import fft
from scipy.stats import entropy
import ROOT

import constants as C
import units as U
import material as mat
import particle as prt
import bins
import fluctuations as flct
import hist
import model

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.15)

ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning

rootpath = "/Users/noamtalhod/tmp/root"
pklpath = "/Users/noamtalhod/tmp/pkl"

################################################################
################################################################
################################################################


def compute_complexity(x, y):
    # 1. Compute curvature
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**(3/2)
    curvature = np.nan_to_num(curvature)  # Handle division by zero
    min_curvature = np.min(curvature)
    max_curvature = np.max(curvature)
    mean_curvature = np.mean(curvature)
    std_curvature = np.std(curvature)
    
    # 2. Fourier spectrum complexity
    y_fft = np.abs(fft(y))
    power_spectrum = y_fft**2
    high_freq_fraction = np.sum(power_spectrum[len(power_spectrum)//4:]) / np.sum(power_spectrum)
    
    # 3. Fractal dimension (box counting method)
    def box_counting(x, y, scales):
        counts = []
        for scale in scales:
            bins = np.arange(min(x), max(x) + scale, scale)
            bin_counts, _ = np.histogram(x, bins)
            counts.append(np.count_nonzero(bin_counts))
        return counts
    
    scales = np.logspace(np.log10(np.min(np.diff(x))), np.log10(np.max(x) - np.min(x)), num=10)
    counts = box_counting(x, y, scales)
    coeffs = np.polyfit(np.log(scales), np.log(counts), 1)
    fractal_dimension = -coeffs[0]
    
    # 4. Total variation
    total_variation = np.sum(np.abs(np.diff(y)))
    
    # 5. Entropy of derivative distribution
    dy_dx = np.gradient(y, x)
    hist, _ = np.histogram(dy_dx, bins=50, density=True)
    hist = hist[hist > 0]  # Remove zero values to avoid log issues
    derivative_entropy = entropy(hist)
    
    return {
        "Avg Curvature": mean_curvature,
        "Std Curvature": std_curvature,
        "High-Frequency Fraction": high_freq_fraction,
        "Fractal Dimension": fractal_dimension,
        "Total Variation": total_variation,
        "Derivative Entropy": derivative_entropy
    }



################################################################
################################################################
################################################################

tf0 = ROOT.TFile("scan_example.root","READ")
href_cnt = tf0.Get("SMALL_h_dL_vs_E")


hnames = ["Avg Curvature",
          "Std Curvature",  
          "High-Frequency Fraction",
          "Fractal Dimension",
          "Total Variation",
          "Derivative Entropy"]
histos = {}
for hname in hnames:
    histos.update({hname:href_cnt.Clone(hname)})
    histos[hname].Reset()
    histos[hname].GetZaxis().SetTitle(hname)



#################################################
#################################################
#################################################
### GENERAL MODEL
dEdxModel  = "G4:Tcut" # or "BB:Tcut"
TargetMat  = mat.Si # or e.g. mat.Al
PrimaryPrt = prt.Particle(name="proton",meV=938.27208816*U.MeV2eV,mamu=1.007276466621,chrg=+1.,spin=0.5,lepn=0,magm=2.79284734463)
par        = flct.Parameters(PrimaryPrt,TargetMat,dEdxModel,"inputs/dEdx_p_si.txt")



############################################################################
nall = href_cnt.GetNbinsX()*href_cnt.GetNbinsY()
nslices = 0
for ie in range(1,href_cnt.GetNbinsX()+1):

    for il in range(1,href_cnt.GetNbinsY()+1):
        
        print(f"Starting slice {nslices+1} out of {nall}")
        
        ### get the slice details
        label = "E"+str(ie)+"_dL"+str(il)
        E     = href_cnt.GetXaxis().GetBinCenter(ie) ## already in MeV
        L     = href_cnt.GetYaxis().GetBinCenter(il) ## already in um
        DE    = href_cnt.GetXaxis().GetBinWidth(ie)
        DL    = href_cnt.GetYaxis().GetBinWidth(il)
        
        ######################################################
        ######################################################
        ######################################################
        ### Build the model shapes
        DOTIME = False
        modelpars  = par.GetModelPars(E*U.MeV2eV,L*U.um2cm)
        Mod = model.Model(L*U.um2cm, E*U.MeV2eV, modelpars, DOTIME)
        # Mod.set_fft_sampling_pars(N_t_bins=10000000,frac=0.05)
        Mod.set_fft_sampling_pars_rotem(N_t_bins=10000000,frac=0.05)
        # Mod.set_all_shapes()
        Mod.set_continuous_shapes()
        # Mod.set_secondaries_shapes()
        x = Mod.cnt_pdfs_arrx
        y = Mod.cnt_pdfs_arrsy["hModel"]

        complexity = compute_complexity(x, y)
        
        for name,cplx in complexity.items(): histos[name].SetBinContent(ie,il,cplx)        
        
        if(nslices%100==0 and nslices!=0): print(f"Processed {nslices} slices out of {nall} total")
        nslices += 1
print(f"\nProcessed {nslices} continuous slices with at least 1 step, out of {nall} total")
#########################################################################################





# gridx,gridy = hist.getGrid(href_cnt)
# for line in gridx: line.SetLineColor(ROOT.kGray)
# for line in gridy: line.SetLineColor(ROOT.kGray)


canvas = ROOT.TCanvas("canvas", "canvas", 500,500)
canvas.cd()
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()
ROOT.gPad.SetLogz()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.18)
href_cnt.GetZaxis().SetTitleOffset(1.6)
href_cnt.Draw("colz")
# for line in gridx: line.Draw("same")
# for line in gridy: line.Draw("same")
ROOT.gPad.RedrawAxis()
canvas.Update()
canvas.SaveAs("test_complexity.pdf(")

for name,hist in histos.items():
    canvas = ROOT.TCanvas("canvas", "canvas", 500,500)
    canvas.cd()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetLeftMargin(0.15)
    ROOT.gPad.SetRightMargin(0.18)
    hist.GetZaxis().SetTitleOffset(1.6)
    hist.Draw("colz")
    # for line in gridx: line.Draw("same")
    # for line in gridy: line.Draw("same")
    ROOT.gPad.RedrawAxis()
    canvas.Update()
    canvas.SaveAs("test_complexity.pdf")

canvas = ROOT.TCanvas("canvas", "canvas", 1000,1000)
canvas.SaveAs("test_complexity.pdf)")



fo = ROOT.TFile("test_complexity.root","RECREATE")
fo.cd()
for name,hist in histos.items(): hist.Write()
fo.Write()
fo.Close()
    
