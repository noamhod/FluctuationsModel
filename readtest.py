import pickle
import math
import array
import numpy as np
import ROOT
from ROOT import TH1D, TH2D, TCanvas, TFile, TLine, TLatex, TLegend
import constants as C
import units as U
import material as mat
import bins
import fluctuations as flct
# import shapes
import rfshapes as rf
import hist

ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.15)


#################################################
#################################################
#################################################
### GENERAL MODEL
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
par = flct.Parameters("Silicon parameters",C.mp,+1,Si,dEdxModel,"inputs/eloss_p_si.txt","inputs/BB.csv")


#################################################
#################################################
#################################################
### specific slice model
XX = 2 ## um
EE = 90 ## MeV
WW = 0.01 ## [%] size of window around XX and EE
model = par.DifferentialModel(EE*U.MeV2eV,XX*U.um2cm,doSec=False)
print(model)
dEmean  = (model["Gauss"]["mean"]+model["Landau"]["mpv"])/2    if("Gauss" in model and "Landau" in model) else model["Landau"]["mpv"]
dEwidth = (model["Gauss"]["width"]+model["Landau"]["width"])/2 if("Gauss" in model and "Landau" in model) else model["Landau"]["width"]
dEmean  *= U.eV2MeV
dEwidth *= U.eV2MeV
dEmin_lin = dEmean-10*dEwidth if((dEmean-10*dEwidth)>0) else 1e-5
dEmax_lin = dEmean+20*dEwidth
print(f"dEmean={dEmean}, dEwidth={dEwidth} --> range:[{dEmin_lin},{dEmax_lin}]")


#################################################
#################################################
#################################################
### general histos:
histos = {}
hist.book(histos)
### slice histos
dEmin = 1e-4
dEmax = 1.
ndEbins,dEbins = bins.GetLogBinning(80,dEmin,dEmax)
slicename  =  f"dE_E{EE}MeV_X{XX}um"
slicetitle = f"E={EE}#pm{WW*100}% [MeV], #Deltax={XX}#pm{WW*100}% [#mum]"
hdE     = TH1D(slicename,slicetitle+";#DeltaE [MeV];Steps",len(dEbins)-1,dEbins)
hdE_cnt = TH1D(slicename+"_cnt",slicetitle+";#DeltaE [MeV];Steps",len(dEbins)-1,dEbins)
hdE_sec = TH1D(slicename+"_sec",slicetitle+";#DeltaE [MeV];Steps",len(dEbins)-1,dEbins)
# hdE_cnt_lin = TH1D(slicename+"_cnt_lin",slicetitle+";#DeltaE [MeV];Steps",1000,1e-4,1e-1)
hdE_cnt_lin = TH1D(slicename+"_cnt_lin",slicetitle+";#DeltaE [MeV];Steps",30,dEmin_lin,dEmax_lin)


#################################################
#################################################
#################################################
# open a file, where you stored the pickled data
fileX = open('data/X.pkl', 'rb')
fileY = open('data/Y.pkl', 'rb')
# dump information to that file
X = pickle.load(fileX)
Y = pickle.load(fileY)
# close the file
fileX.close()
fileY.close()


#################################################
#################################################
#################################################
### Run
for n,enrgy in enumerate(X):
    E     = enrgy*U.eV2MeV
    dx    = Y[n][0]*U.m2um
    dxinv = 1/dx if(dx>0) else -999
    dR    = Y[n][1]*U.m2um
    dRinv = 1/dR if(dR>0) else -999 ## this happens for the primary particles...
    dEcnt = Y[n][2]*U.eV2MeV
    dEtot = Y[n][3]*U.eV2MeV
    dEsec = dEtot-dEcnt
    dE    = dEtot
    Nsec  = int(Y[n][4])
    
    if(E>=bins.Emax):   continue ## skip the primary particles
    if(E<bins.Emin):    continue ## skip the low energy particles
    if(dx>=bins.dxmax): continue ## skip
    if(dx<bins.dxmin):  continue ## skip
        
    # print(f"E={E}: dx={dx}, dR={dR}, dEtot={dEtot}, dEcnt={dEcnt}, dEsec={dEsec}, Nsec={Nsec}")
    
    histos["hdE"].Fill(dE)
    histos["hdE_cnt"].Fill(dEcnt)
    histos["hdE_sec"].Fill(dEsec)
    histos["hdEdx"].Fill(dE/dx)
    histos["hdEdx_cnt"].Fill(dEcnt/dx)
    histos["hdEdx_sec"].Fill(dEsec/dx)
    histos["hdEdx_vs_E"].Fill(E,dE/dx)
    histos["hdEdx_vs_E_small"].Fill(E,dE/dx)
    histos["hdEdx_vs_E_small_cnt"].Fill(E,dEcnt/dx)
    histos["hdEdx_vs_E_small_sec"].Fill(E,dEsec/dx)
    
    if((dx>=(1-WW)*XX and dx<=(1+WW)*XX) and (E>=(1-WW)*EE and E<=(1+WW)*EE)):
        hdE.Fill(dE)
        hdE_sec.Fill(dEsec)
        hdE_cnt.Fill(dEcnt)
        hdE_cnt_lin.Fill(dEcnt)
    
    if(n%1000000==0 and n>0): print("processed: ",n)
    # if(n>1000000): break
print(f"Finished loop")


#################################################
#################################################
#################################################
### 1d dE hists
### 1d dEdx hists
histos["hdE"].SetLineColor(ROOT.kBlack)
histos["hdE_cnt"].SetLineColor(ROOT.kGreen+2)
histos["hdE_sec"].SetLineColor(ROOT.kRed)
histos["hdE"].SetFillColorAlpha(ROOT.kBlack, 0.25)
histos["hdE_cnt"].SetFillColorAlpha(ROOT.kGreen+2, 0.5)
histos["hdE_sec"].SetFillColorAlpha(ROOT.kRed, 0.25)
### 1d dEdx hists
histos["hdEdx"].SetLineColor(ROOT.kBlack)
histos["hdEdx_cnt"].SetLineColor(ROOT.kGreen+2)
histos["hdEdx_sec"].SetLineColor(ROOT.kRed)
histos["hdEdx"].SetFillColorAlpha(ROOT.kBlack, 0.25)
histos["hdEdx_cnt"].SetFillColorAlpha(ROOT.kGreen+2, 0.5)
histos["hdEdx_sec"].SetFillColorAlpha(ROOT.kRed, 0.25)
### slice histos
hdE_sec.Scale(1./hdE_sec.Integral())
hdE_cnt.Scale(1./hdE_cnt.Integral())
hdE.Scale(1./hdE.Integral())
hdE_sec.SetMaximum(0.15)
hdE_cnt.SetMaximum(0.15)
hdE.SetMaximum(0.15)
hdE.SetLineColor(ROOT.kBlack)
hdE_cnt.SetLineColor(ROOT.kGreen+2)
hdE_sec.SetLineColor(ROOT.kRed)
hdE.SetFillColorAlpha(ROOT.kBlack, 0.25)
hdE_cnt.SetFillColorAlpha(ROOT.kGreen+2, 0.5)
hdE_sec.SetFillColorAlpha(ROOT.kRed, 0.25)
### other histos
histos["hdEdx_vs_E_small"].SetTitle("Total")
histos["hdEdx_vs_E_small_cnt"].SetTitle("Continuous only")
histos["hdEdx_vs_E_small_sec"].SetTitle("Secondaries only")
histos["hdEdx_vs_E_small_cnt"].SetMaximum(histos["hdEdx_vs_E_small"].GetMaximum())
histos["hdEdx_vs_E_small_sec"].SetMaximum(histos["hdEdx_vs_E_small"].GetMaximum())
histos["hdEdx_vs_E_small_cnt"].SetMinimum(histos["hdEdx_vs_E_small"].GetMinimum())
histos["hdEdx_vs_E_small_sec"].SetMinimum(histos["hdEdx_vs_E_small"].GetMinimum())
### get the averages
hAv     = hist.getAvgY(histos["hdEdx_vs_E_small"])
hAv_cnt = hist.getAvgY(histos["hdEdx_vs_E_small_cnt"])
hAv_sec = hist.getAvgY(histos["hdEdx_vs_E_small_sec"])
hBB_Tcut = hAv.Clone("BB_Tcut")
hBB_Tmax = hAv.Clone("BB_Tmax")
hBB_G4   = hAv.Clone("BB_G4")
hBB_Tcut.Reset()
hBB_Tcut.Reset()
hBB_G4.Reset()
for b in range(1,hBB_Tcut.GetNbinsX()+1):
    E = hBB_Tcut.GetBinCenter(b)*U.MeV2eV
    dEdx_Tcut = par.BB(E,par.mat.Tc) * (U.eV2MeV/U.cm2um)
    dEdx_Tmax = par.BB(E,par.Wmax(E)) * (U.eV2MeV/U.cm2um)
    dEdx_G4   = par.getG4BBdEdx(E) * (U.eV2MeV/U.cm2um)
    hBB_Tcut.SetBinContent(b,dEdx_Tcut)
    hBB_Tmax.SetBinContent(b,dEdx_Tmax)
    hBB_G4.SetBinContent(b,dEdx_G4)
hAv.SetLineColor(ROOT.kBlack)
hAv_cnt.SetLineColor(ROOT.kBlack)
hAv_sec.SetLineColor(ROOT.kBlack)
hBB_Tcut.SetLineColor(ROOT.kRed)
hBB_Tmax.SetLineColor(ROOT.kGreen+2)
hBB_G4.SetLineColor(ROOT.kViolet)
hBB_G4.SetLineStyle(ROOT.kDotted)




#################################################
#################################################
#################################################
### plot everything
pdf = "readtest.pdf"

cnv = TCanvas("cnv","",500,500)
ROOT.gPad.SetLogy()
ROOT.gPad.SetLogx()
ROOT.gPad.SetTicks(1,1)
histos["hdEdx"].Draw("hist")
histos["hdEdx_sec"].Draw("hist same")
histos["hdEdx_cnt"].Draw("hist same")
leg = TLegend(0.5,0.7,0.8,0.88)
leg.SetFillStyle(4000) # will be transparent
leg.SetFillColor(0)
leg.SetTextFont(42)
leg.SetBorderSize(0)
leg.AddEntry(histos["hdEdx"],"Total","f")
leg.AddEntry(histos["hdEdx_sec"],"Secondaries","f")
leg.AddEntry(histos["hdEdx_cnt"],"Continuous","f")
leg.Draw("same")
ROOT.gPad.RedrawAxis()
cnv.SaveAs(pdf+"(")

cnv = TCanvas("cnv","",500,500)
ROOT.gPad.SetLogy()
ROOT.gPad.SetLogx()
ROOT.gPad.SetTicks(1,1)
histos["hdE"].Draw("hist")
histos["hdE_sec"].Draw("hist same")
histos["hdE_cnt"].Draw("hist same")
leg = TLegend(0.15,0.7,0.45,0.88)
leg.SetFillStyle(4000) # will be transparent
leg.SetFillColor(0)
leg.SetTextFont(42)
leg.SetBorderSize(0)
leg.AddEntry(histos["hdE"],"Total","f")
leg.AddEntry(histos["hdE_sec"],"Secondaries","f")
leg.AddEntry(histos["hdE_cnt"],"Continuous","f")
leg.Draw("same")
ROOT.gPad.RedrawAxis()
cnv.SaveAs(pdf)

cnv = TCanvas("cnv","",1500,500)
leg = TLegend(0.5,0.7,0.8,0.88)
leg.SetFillStyle(4000) # will be transparent
leg.SetFillColor(0)
leg.SetTextFont(42)
leg.SetBorderSize(0)
leg.AddEntry(hAv,"Avg of 2D hist","l")
leg.AddEntry(hBB_Tcut,"BB with T_{cut}","l")
leg.AddEntry(hBB_Tmax,"BB with T_{max}","l")
leg.AddEntry(hBB_G4,"BB from G4","l")
cnv.Divide(3,1)
cnv.cd(1)
histos["hdEdx_vs_E_small"].Draw("colz")
hBB_Tcut.Draw("hist same")
hBB_Tmax.Draw("hist same")
hBB_G4.Draw("hist same")
hAv.Draw("hist same")
leg.Draw("same")
ROOT.gPad.SetLogy()
ROOT.gPad.SetLogz()
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.RedrawAxis()
cnv.cd(2)
histos["hdEdx_vs_E_small_cnt"].Draw("colz")
hBB_Tcut.Draw("hist same")
hBB_Tmax.Draw("hist same")
hBB_G4.Draw("hist same")
hAv_cnt.Draw("hist same")
leg.Draw("same")
ROOT.gPad.SetLogy()
ROOT.gPad.SetLogz()
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.RedrawAxis()
cnv.cd(3)
histos["hdEdx_vs_E_small_sec"].Draw("colz")
hBB_Tcut.Draw("hist same")
hBB_Tmax.Draw("hist same")
hBB_G4.Draw("hist same")
hAv_sec.Draw("hist same")
leg.Draw("same")
ROOT.gPad.SetLogy()
ROOT.gPad.SetLogz()
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.RedrawAxis()
cnv.SaveAs(pdf)


cnv = TCanvas("cnv","",500,500)
ROOT.gPad.SetLogx()
ROOT.gPad.SetTicks(1,1)
hdE.Draw("hist")
hdE_sec.Draw("hist same")
hdE_cnt.Draw("hist same")
leg = TLegend(0.5,0.7,0.8,0.88)
leg.SetFillStyle(4000) # will be transparent
leg.SetFillColor(0)
leg.SetTextFont(42)
leg.SetBorderSize(0)
leg.AddEntry(hdE,"Total","f")
leg.AddEntry(hdE_sec,"Secondaries","f")
leg.AddEntry(hdE_cnt,"Continuous","f")
leg.Draw("same")
ROOT.gPad.RedrawAxis()
cnv.SaveAs(pdf)


###################
### RooFit shapes
rfs = rf.RooFitShapes("roofitshapes",model,hdE_cnt_lin)
frame = rfs.plot()
###################
### final plot
canvas = ROOT.TCanvas("canvas", "canvas", 100, 100, 800, 600)
legend = ROOT.TLegend(0.2, 0.6, 0.4, 0.85)
legend.SetTextSize(0.032)
legend.SetBorderSize(0)
legend.SetFillStyle(0)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
# ROOT.gPad.SetLogx()
# ROOT.gPad.SetLogy()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.02)
frame.GetYaxis().SetLabelOffset(0.008)
frame.GetYaxis().SetTitleOffset(1.5)
frame.GetYaxis().SetTitleSize(0.045)
frame.GetXaxis().SetTitleSize(0.045)
frame.Draw()
legend.AddEntry("data", "data", 'LEP')
if(rfs.isConv):   legend.AddEntry("full_model", "Convolution", 'L')
if(rfs.isLandau): legend.AddEntry("landau_model", "Landau", 'L')
if(rfs.isGauss):  legend.AddEntry("gauss_model", "Gauss", 'L')
legend.Draw()
canvas.SaveAs(pdf+")")





