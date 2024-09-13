import ROOT
from array import array
import constants as C
import units as U
import material as mat
import fluctuations as fluct

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.15)


Emin = 1.e-5 #0.1  # MeV
Emax = 100 # MeV

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
par = fluct.Parameters("Silicon parameters",C.mp,+1,Si,dEdxModel,"inputs/eloss_p_si.txt","inputs/BB.csv")

def getBB_GEANT4(fname):
    hname = fname
    hname = hname.split(".")[0]
    arr_E        = array( 'd' )
    arr_dEdx_low = array( 'd' )
    arr_dEdx_std = array( 'd' )
    with open(fname) as f:
        for line in f:
            if("#" in line): continue
            words = line.split(",")
            if(float(words[0])<Emin): continue
            if(float(words[0])>Emax): break
            
            print(f"E=",float(words[0]))
            
            arr_E.append( float(words[0]) )
            arr_dEdx_low.append( float(words[1]) )
            arr_dEdx_std.append( float(words[2]) )
    npts = len(arr_E)
    
    gBBlow = ROOT.TGraph(npts,arr_E,arr_dEdx_low)
    gBBlow.SetLineColor(ROOT.kRed)
    gBBlow.GetXaxis().SetTitle("E [MeV]")
    gBBlow.GetYaxis().SetTitle("dE/dx [MeV/mm]")
    
    gBBstd = ROOT.TGraph(npts,arr_E,arr_dEdx_std)
    gBBstd.SetLineColor(ROOT.kBlack)
    gBBstd.GetXaxis().SetTitle("E [MeV]")
    gBBstd.GetYaxis().SetTitle("dE/dx [MeV/mm]")

    return gBBlow,gBBstd


def setG4BBdEdxFromTable(fname):
    hname = fname
    hname = hname.split(".")[0]
    arr_E    = array( 'd' )
    arr_dEdx = array( 'd' )
    with open(fname) as f:
        for line in f:
            if("#" in line): continue
            line = line.replace("\n","")
            words = line.split("   ")
            arr_E.append( float(words[0]) )
            arr_dEdx.append( float(words[1]) )
    npts = len(arr_E)
    # print(f"Read {npts} points from file {fname}")
    gBB = ROOT.TGraph(npts,arr_E,arr_dEdx)
    gBB.SetLineColor(ROOT.kBlue)
    gBB.GetXaxis().SetTitle("E [MeV]")
    gBB.GetYaxis().SetTitle("dE/dx [MeV/mm]")
    return gBB


def getBB_PDG(model="std"):
    arr_E         = array( 'd' )
    arr_dEdx_Tmax = array( 'd' )
    arr_dEdx_Tcut = array( 'd' )
    arr_dEdx_Tup  = array( 'd' )
    arr_rsec = array( 'd' )
    npts  = 1000
    Estep = (Emax-Emin)/npts
    for i in range(npts):
        E = Emin + i*Estep
        Tmax = par.Wmax(E*U.MeV2eV)
        Tcut = par.mat.Tc
        Tup  = par.Tup(E*U.MeV2eV)
        BBmax = par.BB(E*U.MeV2eV,Tmax)*(U.eV2MeV/U.cm2mm) if(model=="std") else par.BBlowE(E*U.MeV2eV,Tmax)*(U.eV2MeV/U.cm2mm)
        BBcut = par.BB(E*U.MeV2eV,Tcut)*(U.eV2MeV/U.cm2mm) if(model=="std") else par.BBlowE(E*U.MeV2eV,Tcut)*(U.eV2MeV/U.cm2mm)
        BBup  = par.BB(E*U.MeV2eV,Tup)*(U.eV2MeV/U.cm2mm)  if(model=="std") else par.BBlowE(E*U.MeV2eV,Tup)*(U.eV2MeV/U.cm2mm)
        rsec  = (BBmax-BBcut)/BBmax
        # BB = par.BBlowE(E*U.MeV2eV,"cut")*U.eV2MeV*(1/U.cm2mm)
        arr_E.append( E )
        arr_dEdx_Tmax.append( BBmax )
        arr_dEdx_Tcut.append( BBcut )
        arr_dEdx_Tup.append( BBup )
        arr_rsec.append( rsec )
    
    gBBTmax = ROOT.TGraph(npts,arr_E,arr_dEdx_Tmax)
    gBBTmax.SetLineColor(ROOT.kGreen+2)
    gBBTmax.GetXaxis().SetTitle("E [MeV]")
    gBBTmax.GetYaxis().SetTitle("dE/dx [MeV/mm]")
    
    gBBTcut = ROOT.TGraph(npts,arr_E,arr_dEdx_Tcut)
    gBBTcut.SetLineColor(ROOT.kMagenta)
    gBBTcut.GetXaxis().SetTitle("E [MeV]")
    gBBTcut.GetYaxis().SetTitle("dE/dx [MeV/mm]")
    
    gBBTup = ROOT.TGraph(npts,arr_E,arr_dEdx_Tup)
    gBBTup.SetLineColor(ROOT.kBlue)
    gBBTup.GetXaxis().SetTitle("E [MeV]")
    gBBTup.GetYaxis().SetTitle("dE/dx [MeV/mm]")
    
    gRsec = ROOT.TGraph(npts,arr_E,arr_rsec)
    gRsec.SetLineColor(ROOT.kBlack)
    gRsec.GetXaxis().SetTitle("E [MeV]")
    gRsec.GetYaxis().SetTitle("r_{sel}")
    
    return gBBTmax,gBBTcut,gBBTup,gRsec

### from PDG (theory)
gBBTmax,gBBTcut,gBBTup,gRsec = getBB_PDG("std")
# gBBlowTmax,gBBlowTcut,glowRsec = getBB_PDG("low")
### from GEANT4 directly
# gBBlow,gBBstd = getBB_GEANT4("inputs/BB.csv")
gBBstd = setG4BBdEdxFromTable("inputs/eloss_p_si.txt")

leg = ROOT.TLegend(0.5,0.74,0.8,0.88)
leg.SetFillStyle(4000) # will be transparent
leg.SetFillColor(0)
leg.SetTextFont(42)
leg.SetBorderSize(0)
leg.AddEntry(gBBstd,"BB(GEANT4,std) with T_{cut}","l")
leg.AddEntry(gBBTmax,"BB(PDG,std) with T_{max}","l")
leg.AddEntry(gBBTcut,"BB(PDG,std) with T_{cut}","l")
leg.AddEntry(gBBTup,"BB(PDG,std) with T_{up}","l")


cnv = ROOT.TCanvas("cnv","",500,500)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
mg = ROOT.TMultiGraph()
mg.Add(gBBstd)
mg.Add(gBBTmax)
mg.Add(gBBTcut)
mg.Add(gBBTup)
# mg.Add(gBBlowTmax)
# mg.Add(gBBlowTcut)
mg.GetXaxis().SetTitle( gBBstd.GetXaxis().GetTitle() )
mg.GetYaxis().SetTitle( gBBstd.GetYaxis().GetTitle() )
mg.Draw("al")
leg.Draw("same")
ROOT.gPad.RedrawAxis()
cnv.SaveAs("dEdx_cpp.pdf(")


cnv = ROOT.TCanvas("cnv","",500,500)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogx()
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
# gRsec.SetMinimum(0)
gRsec.SetTitle("")
gRsec.Draw("ac")
ROOT.gPad.RedrawAxis()
cnv.SaveAs("dEdx_cpp.pdf")


arr_E      = array( 'd' )
arr_Sigma1 = array( 'd' )
arr_Sigma2 = array( 'd' )
arr_Sigma3 = array( 'd' )
arr_Sigma4 = array( 'd' )
arr_Sigma0 = array( 'd' )
nsteps = 10000
Estep = (Emax-Emin)/nsteps
for i in range(nsteps):
    E = Emin + i*Estep
    arr_E.append( E )
    arr_Sigma1.append( par.Sigma12(E*U.MeV2eV,1) )
    arr_Sigma2.append( par.Sigma12(E*U.MeV2eV,2) )
    arr_Sigma3.append( par.Sigma3(E*U.MeV2eV) )
    arr_Sigma4.append( par.Sigma4(E*U.MeV2eV) )
    arr_Sigma0.append( par.Sigma0(E*U.MeV2eV) )
npts = len(arr_E)
gSig1 = ROOT.TGraph(npts,arr_E,arr_Sigma1)
gSig1.SetLineColor(ROOT.kBlack)
gSig1.GetXaxis().SetTitle("E [MeV]")
gSig1.GetYaxis().SetTitle("#Sigma_{x} [1/cm]")
gSig2 = ROOT.TGraph(npts,arr_E,arr_Sigma2)
gSig2.SetLineColor(ROOT.kRed)
gSig2.GetXaxis().SetTitle("E [MeV]")
gSig2.GetYaxis().SetTitle("#Sigma_{x} [1/cm]")
gSig3 = ROOT.TGraph(npts,arr_E,arr_Sigma3)
gSig3.SetLineColor(ROOT.kGreen)
gSig3.GetXaxis().SetTitle("E [MeV]")
gSig3.GetYaxis().SetTitle("#Sigma_{x} [1/cm]")
gSig4 = ROOT.TGraph(npts,arr_E,arr_Sigma4)
gSig4.SetLineColor(ROOT.kBlue)
gSig4.GetXaxis().SetTitle("E [MeV]")
gSig4.GetYaxis().SetTitle("#Sigma_{x} [1/cm]")
gSig0 = ROOT.TGraph(npts,arr_E,arr_Sigma0)
gSig0.SetLineColor(ROOT.kOrange)
gSig0.GetXaxis().SetTitle("E [MeV]")
gSig0.GetYaxis().SetTitle("#Sigma_{x} [1/cm]")
cnv = ROOT.TCanvas("cnv","",500,500)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()
ROOT.gPad.SetGridx()
ROOT.gPad.SetGridy()
mg = ROOT.TMultiGraph()
mg.Add(gSig1)
mg.Add(gSig2)
mg.Add(gSig3)
mg.Add(gSig4)
mg.Add(gSig0)
mg.GetXaxis().SetTitle( gSig1.GetXaxis().GetTitle() )
mg.GetYaxis().SetTitle( gSig1.GetYaxis().GetTitle() )
mg.Draw("al")
leg = ROOT.TLegend(0.5,0.74,0.8,0.88)
leg.SetFillStyle(4000) # will be transparent
leg.SetFillColor(0)
leg.SetTextFont(42)
leg.SetBorderSize(0)
leg.AddEntry(gSig1,"#Sigma_{1}","l")
leg.AddEntry(gSig2,"#Sigma_{2}","l")
leg.AddEntry(gSig3,"#Sigma_{3}","l")
leg.AddEntry(gSig4,"#Sigma_{4}","l")
leg.AddEntry(gSig0,"#Sigma_{0}","l")
leg.Draw("same")
ROOT.gPad.RedrawAxis()
cnv.SaveAs("dEdx_cpp.pdf)")


# nsteps = 100
# Estep = (Emax-Emin)/nsteps
# X = 1*U.um2cm
# for i in range(nsteps):
#     E = Emin + i*Estep
#     Tmax = par.Wmax(E*U.MeV2eV)*U.eV2MeV
#     # print(f"is sec for E={E} [MeV], X={X} [cm], Tmax={Tmax} : {par.isSecondary(E*U.MeV2eV,X)}")
    