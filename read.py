import pickle
import math
import array
import numpy as np
import ROOT
from ROOT import TH1D, TH2D, TCanvas, TFile, TLine, TLatex
import units as U
import constants as C
import material as mat
import bins
import fluctuations as flct
import shapes
import hist

ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.15)


# open a file, where you stored the pickled data
fileX = open('data/X.pkl', 'rb')
fileY = open('data/Y.pkl', 'rb')

# dump information to that file
X = pickle.load(fileX)
Y = pickle.load(fileY)

# close the file
fileX.close()
fileY.close()

### histos:
histos = {}
hist.book(histos)

### Run
for n,enrgy in enumerate(X):
    # E     = enrgy*U.eV2MeV
    # dx    = Y[n][0]*U.m2um
    # dxinv = 1/dx if(dx>0) else -999
    # dE    = 1*Y[n][1]*U.eV2MeV
    # dR    = Y[n][2]*U.m2um
    # dRinv = 1/dR if(dR>0) else -999 ## this happens for the primary particles...
    # if(E>=histos["hE"].GetXaxis().GetXmax()): continue ## skip the primary particles
    # if(E<histos["hE"].GetXaxis().GetXmin()):  continue ## skip the low energy particles
    # if(dx>=histos["hdx"].GetXaxis().GetXmax()): continue ## skip
    # if(dx<histos["hdx"].GetXaxis().GetXmin()):  continue ## skip
    
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
    
    ie = histos["SMALL_hdx_vs_E"].GetXaxis().FindBin(E)
    ix = histos["SMALL_hdx_vs_E"].GetYaxis().FindBin(dx)
    label = "E"+str(ie)+"_dx"+str(ix)
    histos["hdE_"+label].Fill(dE)
    histos["hE_"+label].Fill(E)
    histos["hdx_"+label].Fill(dx)
    
    ie    = histos["SMALL_hdxinv_vs_E"].GetXaxis().FindBin(E)
    ixinv = histos["SMALL_hdxinv_vs_E"].GetYaxis().FindBin(dxinv)
    # print(f"E={E}, dx={dx}, 1/dx={dxinv}  -->  ie={ie}, ixinv={ixinv}")
    
    label = "E"+str(ie)+"_dxinv"+str(ixinv)
    histos["hdE_"+label].Fill(dE)
    histos["hE_"+label].Fill(E)
    histos["hdxinv_"+label].Fill(dxinv)
    histos["hdx_"+label].Fill(dx)
    # print(f'{n}: label={label} --> rangeE[{histos["SMALL_hdxinv_vs_E"].GetXaxis().GetBinLowEdge(ie)},{histos["SMALL_hdxinv_vs_E"].GetXaxis().GetBinUpEdge(ie)}] --> E={E} [MeV] -->  range1/x[{histos["SMALL_hdxinv_vs_E"].GetYaxis().GetBinLowEdge(ixinv)},{histos["SMALL_hdxinv_vs_E"].GetYaxis().GetBinUpEdge(ixinv)}]  --> 1/dx={dxinv}, dE={dE} [MeV]')
    
    if(n%1000000==0 and n>0): print("processed: ",n)
    if(n>1000000): break


pdf = "out.pdf"

cnv = TCanvas("cnv","",1000,1000)
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

cnv = TCanvas("cnv","",1000,1000)
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

cnv = TCanvas("cnv","",1000,500)
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

cnv = TCanvas("cnv","",1000,500)
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

cnv = TCanvas("cnv","",500,500)
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

cnv = TCanvas("cnv","",500,500)
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


cnv = TCanvas("cnv","",500,500)
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

cnv = TCanvas("cnv","",500,500)
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
cnv.SaveAs(pdf)


### first normalize
hmin_dE, hmax_dE      = hist.hNorm(histos,"SMALL_hdxinv_vs_E","E","dxinv","dE")
hmin_E,hmax_E         = hist.hNorm(histos,"SMALL_hdxinv_vs_E","E","dxinv","E")
hmin_dxinv,hmax_dxinv = hist.hNorm(histos,"SMALL_hdxinv_vs_E","E","dxinv","dxinv")
hmin_dx,hmax_dx       = hist.hNorm(histos,"SMALL_hdxinv_vs_E","E","dx","dx")


rho_Al = 2.699     # Aluminum, g/cm3
Z_Al   = [13]      # Aluminum atomic number (Z)
A_Al   = [26.98]   # Aluminum atomic mass (A)
I_Al   = 166.0     # Aluminum mean excitation energy (I), eV
Ep_Al  = 32.86     # Aluminum plasma energy (E_p), eV
Tc_Al  = 990       # Aluminum production threshold for delta ray, eV
den_Al = [32.86, 2.18, 4.2395, 0.1708, 3.0127, 0.08024, 3.6345, 0.12, 0.061, 166.]
nel_Al = 1
Al = mat.Material("Aluminum","Al",rho_Al,Z_Al,A_Al,I_Al,Tc_Al,den_Al,nel_Al)

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
func = shapes.Functions("Landau")


test0 = TH1D("test0",";Theory_{#sigma}/Hist_{#sigma}",100,0,2)
test1 = TH1D("test1",";Theory_{MPV}/Hist_{MPV}",100,0,2)
test2 = TH1D("test2",";Theory_{Mean}/Hist_{Mean}",100,0,2)
test3 = TH1D("test3",";Theory_{MPV}/PDG_{MPV}",100,0,2)


### make gif for all bins
ROOT.gSystem.Unlink("out.gif") ## remove old files
ROOT.gSystem.Exec("/bin/rm -f out.gif") ## remove old files
ROOT.gSystem.Exec("/bin/rm -rf /Users/noamtalhod/tmp/png") ## remove old files
ROOT.gSystem.Exec("/bin/mkdir -p /Users/noamtalhod/tmp/png")
NminRawSteps = 25
count = 0
for ie in range(1,histos["SMALL_hdxinv_vs_E"].GetNbinsX()+1):
    label_E = str(ie)
    for ixinv in range(1,histos["SMALL_hdxinv_vs_E"].GetNbinsY()+1):
        label_dxinv = str(ixinv)
        label = "E"+label_E+"_dxinv"+label_dxinv
        name = "hdE_"+label
        NrawSteps = histos["SMALL_hdxinv_vs_E"].GetBinContent(ie,ixinv)

        ### get the E and x before skippong ay E-X bin
        midRangeE = (histos["hE_"+label].GetXaxis().GetXmax()-histos["hE_"+label].GetXaxis().GetXmin())/2.
        midRangeX = (histos["hdx_"+label].GetXaxis().GetXmax()-histos["hdx_"+label].GetXaxis().GetXmin())/2.
        E = histos["hE_"+label].GetMean()*U.MeV2eV if(NrawSteps>=NminRawSteps) else midRangeE*U.MeV2eV # eV
        x = histos["hdx_"+label].GetMean()*U.um2cm if(NrawSteps>=NminRawSteps) else midRangeX*U.um2cm  # cm

        ### before skipping any E-x bins
        binNi = histos["hN1_E"+label_E].FindBin(x*U.cm2um)
        # print(f"label={label}, E={E*U.eV2MeV}, x={x*U.cm2um}, binNi={binNi}")
        histos["hN1_E"+label_E].SetBinContent(binNi, par.n12_mean(E,x,1))
        histos["hN3_E"+label_E].SetBinContent(binNi, par.n3_mean(E,x))
        histos["hN0_E"+label_E].SetBinContent(binNi, par.n_0dE_mean(E,x))

        ### skip E-x bin if there are too few raw steps
        if(NrawSteps<10): continue
        
        cgif = TCanvas("gif","",1000,1000)
        cgif.Divide(2,2)
        cgif.cd(1)
        ROOT.gPad.SetLogx()
        # ROOT.gPad.SetLogy()
        ROOT.gPad.SetTicks(1,1)
        dEmaximum = histos[name].GetBinCenter( histos[name].GetMaximumBin() )
        hdEmaximum = histos[name].GetBinContent( histos[name].GetMaximumBin() )
        # histos[name].SetMinimum(hmin_dE)
        # histos[name].SetMaximum(hmax_dE)
        histos[name].Draw("hist")
        
        dEmin = histos[name].GetXaxis().GetXmin()
        dEmax = histos[name].GetXaxis().GetXmax()
        
        mean_dEdx_hist = histos[name].GetMean() ## akready in MeV
        sigma_dEdx_hist = histos[name].GetStdDev() ## akready in MeV
        # mean_dEdx_bbpdg = x*(par.BB(E,par.Wmax(E))*(1-par.Rsec(E)))*U.eV2MeV  # x*par.getG4BBdEdx(E)*U.eV2MeV
        mean_dEdx_bbpdg = x*par.BB(E,par.Wmax(E))*U.eV2MeV  # x*par.getG4BBdEdx(E)*U.eV2MeV
        mean_dEdx_modl = par.Mean(E,x)*U.eV2MeV
        mean_dEdx = par.Mean(E,x)*U.eV2MeV
        Delta_p,Width,Model = par.Model(E,x)
        Delta_p = Delta_p*U.eV2MeV
        Width   = Width*U.eV2MeV
        f = None
        if(Model=="Gaus"):   f = func.fGaus(dEmin,dEmax,[Delta_p, Width, 1],label)
        if(Model=="Landau"): f = func.fLandau(dEmin,dEmax,[Delta_p, Width, 1],label)
        
        # f = func.fLandau(dEmin,dEmax,[Delta_p, Width, 1],label)
        h = func.f2h(f,histos[name])
        hmin_f2h,hmax_f2h = hist.getH1minmax(h)
        h.Scale(hdEmaximum/hmax_f2h)
        h.SetFillColorAlpha(h.GetLineColor(),0.30)
        # f.Draw("same")
        h.Draw("hist same")
        histos[name].Draw("hist same") ## redraw the hist on top
        ROOT.gPad.RedrawAxis()
        
        Widthratio = Width/sigma_dEdx_hist if(sigma_dEdx_hist>0) else -999
        MPVratio = Delta_p/dEmaximum
        MVPratio_PDG = Delta_p/(par.Delta_p_PDG(E,x)*U.eV2MeV)
        Meanratio = mean_dEdx_bbpdg/mean_dEdx_hist
        # print(f'{label}: E={E*U.eV2MeV} [MeV], x={x*U.cm2um} [um] --> Delta_p={Delta_p}, Width={Width}, Delta_p/hXmax:{Delta_p/dEmaximum} N={histos[name].GetEntries()}')
        if(histos[name].GetEntries()>10):
            test0.Fill(Widthratio)
            test1.Fill(MPVratio)
            test2.Fill(Meanratio)
            test3.Fill(MVPratio_PDG)
            binR1 = histos["hMPVratio_E"+label_E].FindBin(x*U.cm2um)
            histos["hMPVratio_E"+label_E].SetBinContent(binR1,MPVratio)
            binR2 = histos["hMeanratio_E"+label_E].FindBin(x*U.cm2um)
            histos["hMeanratio_E"+label_E].SetBinContent(binR2,Meanratio)
        
        cgif.cd(2)
        name = "hE_"+label
        ROOT.gPad.SetLogy()
        ROOT.gPad.SetTicks(1,1)
        histos[name].SetMinimum(hmin_E)
        histos[name].SetMaximum(hmax_E)
        histos[name].Draw("hist")
        ROOT.gPad.RedrawAxis()
        ROOT.gPad.Update()

        cgif.cd(3)
        # name = "hdxinv_"+label
        # ROOT.gPad.SetLogy()
        # ROOT.gPad.SetTicks(1,1)
        # # print(f'Integral of {label} is {histos[name].Integral("width")}')
        # histos[name].SetMinimum(hmin_dxinv)
        # histos[name].SetMaximum(hmax_dxinv)
        # histos[name].Draw("hist")
        # ROOT.gPad.RedrawAxis()

        ### the text
        s = ROOT.TLatex()
        s.SetNDC(1);
        s.SetTextAlign(13);
        s.SetTextFont(22);
        s.SetTextColor(ROOT.kBlack)
        s.SetTextSize(0.04)
        s.DrawLatex(0.3,0.90,ROOT.Form("E #in [%.3e, %.3e) [MeV]" % (histos["hE_"+label].GetXaxis().GetXmin(), histos["hE_"+label].GetXaxis().GetXmax())))
        s.DrawLatex(0.3,0.84,ROOT.Form("dx #in [%.3e, %.3e) [#mum]" % (histos["hdx_"+label].GetXaxis().GetXmin(), histos["hdx_"+label].GetXaxis().GetXmax())))
        s.DrawLatex(0.3,0.78,ROOT.Form("N raw steps = %d" % (NrawSteps)))
        s.DrawLatex(0.3,0.72,ROOT.Form("E_{mean} = %.3e [MeV]" % (E*U.eV2MeV)))
        s.DrawLatex(0.3,0.66,ROOT.Form("dx_{mean} = %.3e [#mum]" % (x*U.cm2um)))
        s.DrawLatex(0.3,0.60,ROOT.Form("MPV_{model} = %.3e [MeV]" % (Delta_p)))
        s.DrawLatex(0.3,0.54,ROOT.Form("#sigma_{model} = %.3e [MeV]" % (Width)))
        s.DrawLatex(0.3,0.48,ROOT.Form("#sigma_{data} = %.3e [MeV]" % (sigma_dEdx_hist)))
        s.DrawLatex(0.3,0.42,ROOT.Form("<dE>_{data} = %.3e [MeV]" % (mean_dEdx_hist)))
        s.DrawLatex(0.3,0.36,ROOT.Form("<dE>_{pdg} = %.3e [MeV]" % (mean_dEdx_bbpdg)))
        s.DrawLatex(0.3,0.30,ROOT.Form("<dE>_{model} = %.3e [MeV]" % (mean_dEdx_modl)))
        ROOT.gPad.RedrawAxis()
        ROOT.gPad.Update()
        
        cgif.cd(4)
        name = "hdx_"+label
        ROOT.gPad.SetLogy()
        ROOT.gPad.SetTicks(1,1)
        # print(f'Integral of {label} is {histos[name].Integral("width")}')
        histos[name].SetMinimum(hmin_dx)
        histos[name].SetMaximum(hmax_dx)
        histos[name].Draw("hist")
        ROOT.gPad.RedrawAxis()
        ROOT.gPad.Update()
        
        cgif.Update()
        cgif.Print(f"/Users/noamtalhod/tmp/png/out_{count}.png")
        count += 1
print("Making gif...")
ROOT.gSystem.Exec("convert -delay 20 $(ls /Users/noamtalhod/tmp/png/*.png | sort -V) out.gif")


pdfdeltapvsdx = "Ratios_vs_dx.pdf"
cnv = TCanvas("cnv","",500,500)
cnv.SaveAs(pdfdeltapvsdx+"(")
for ie in range(1,histos["SMALL_hdxinv_vs_E"].GetNbinsX()+1):
    label_E = str(ie)
    cnv = TCanvas("cnv","",500,500)
    cnv.Divide(2,2)
    
    cnv.cd(1)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetTicks(1,1)
    histos["hMPVratio_E"+label_E].SetMinimum(0)
    histos["hMPVratio_E"+label_E].SetMaximum(10)
    histos["hMPVratio_E"+label_E].Draw("hist")    
    ### the text
    s = ROOT.TLatex()
    s.SetNDC(1);
    s.SetTextAlign(13);
    s.SetTextFont(22);
    s.SetTextColor(ROOT.kBlack)
    s.SetTextSize(0.03)
    s.DrawLatex(0.4,0.85,ROOT.Form("E #in [%.3e, %.3e) [MeV]" % (histos["SMALL_hdxinv_vs_E"].GetXaxis().GetBinLowEdge(ie), histos["SMALL_hdxinv_vs_E"].GetXaxis().GetBinUpEdge(ie))))
    ROOT.gPad.RedrawAxis()
    
    cnv.cd(2)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetTicks(1,1)
    histos["hMeanratio_E"+label_E].SetMinimum(0)
    histos["hMeanratio_E"+label_E].SetMaximum(2)
    histos["hMeanratio_E"+label_E].Draw("hist")
    s.DrawLatex(0.4,0.85,ROOT.Form("E #in [%.3e, %.3e) [MeV]" % (histos["SMALL_hdxinv_vs_E"].GetXaxis().GetBinLowEdge(ie), histos["SMALL_hdxinv_vs_E"].GetXaxis().GetBinUpEdge(ie))))
    ROOT.gPad.RedrawAxis()
    
    cnv.cd(3)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    hmaxs = []
    hmaxs.append( histos["hN1_E"+label_E].GetMaximum() )
    hmaxs.append( histos["hN3_E"+label_E].GetMaximum() )
    hmaxs.append( histos["hN0_E"+label_E].GetMaximum() )
    hmax = -1e20
    for m in hmaxs:
        hmax = m if(m>hmax) else hmax
    hmins = []
    hmins.append( histos["hN1_E"+label_E].GetMinimum() )
    hmins.append( histos["hN3_E"+label_E].GetMinimum() )
    hmins.append( histos["hN0_E"+label_E].GetMinimum() )
    hmin = +1e20
    for m in hmins:
        hmin = m if(m<hmin) else hmin
    hmin = 0.5 if(hmin==0) else hmin
    histos["hN1_E"+label_E].SetMaximum(hmax*2.0)
    histos["hN3_E"+label_E].SetMaximum(hmax*2.0)
    histos["hN0_E"+label_E].SetMaximum(hmax*2.0)
    histos["hN1_E"+label_E].SetMinimum(hmin*0.5)
    histos["hN3_E"+label_E].SetMinimum(hmin*0.5)
    histos["hN0_E"+label_E].SetMinimum(hmin*0.5)
    histos["hN1_E"+label_E].SetLineColor(ROOT.kBlack)
    histos["hN3_E"+label_E].SetLineColor(ROOT.kRed)
    histos["hN0_E"+label_E].SetLineColor(ROOT.kGreen)
    histos["hN1_E"+label_E].Draw("hist")
    histos["hN3_E"+label_E].Draw("hist same")
    histos["hN0_E"+label_E].Draw("hist same")
    s.DrawLatex(0.4,0.85,ROOT.Form("E #in [%.3e, %.3e) [MeV]" % (histos["SMALL_hdxinv_vs_E"].GetXaxis().GetBinLowEdge(ie), histos["SMALL_hdxinv_vs_E"].GetXaxis().GetBinUpEdge(ie))))
    ROOT.gPad.RedrawAxis()
    
    cnv.SaveAs(pdfdeltapvsdx)
cnv = TCanvas("cnv","",500,500)
cnv.SaveAs(pdfdeltapvsdx+")")    


for ie in range(1,histos["SMALL_hdx_vs_E_isGauss_N1"].GetNbinsX()+1):
    E = U.MeV2eV * histos["SMALL_hdx_vs_E_isGauss_N1"].GetXaxis().GetBinCenter(ie)
    for ix in range(1,histos["SMALL_hdx_vs_E_isGauss_N1"].GetNbinsY()+1):
        x = U.um2cm * histos["SMALL_hdx_vs_E_isGauss_N1"].GetYaxis().GetBinCenter(ix)
        isG1 = 1 if(par.isGauss(E,x,1)) else 1e-6
        isG3 = 1 if(par.isGauss(E,x,3)) else 1e-6
        isG0 = 1 if(par.isGauss(E,x,0)) else 1e-6
        n1 = par.n12_mean(E,x,1)
        n3 = par.n3_mean(E,x)
        n0 = par.n_0dE_mean(E,x)
        histos["SMALL_hdx_vs_E_isGauss_N1"].SetBinContent(ie,ix,isG1)
        histos["SMALL_hdx_vs_E_isGauss_N3"].SetBinContent(ie,ix,isG3)
        histos["SMALL_hdx_vs_E_isGauss_N0"].SetBinContent(ie,ix,isG0)
        histos["SMALL_hdx_vs_E_N1"].SetBinContent(ie,ix,n1)
        histos["SMALL_hdx_vs_E_N3"].SetBinContent(ie,ix,n3)
        histos["SMALL_hdx_vs_E_N0"].SetBinContent(ie,ix,n0)
histos["SMALL_hdx_vs_E_isGauss_N1"].SetMinimum(0)
histos["SMALL_hdx_vs_E_isGauss_N3"].SetMinimum(0)
histos["SMALL_hdx_vs_E_isGauss_N0"].SetMinimum(0)
histos["SMALL_hdx_vs_E_isGauss_N1"].SetMaximum(1)
histos["SMALL_hdx_vs_E_isGauss_N3"].SetMaximum(1)
histos["SMALL_hdx_vs_E_isGauss_N0"].SetMaximum(1)


cnv = TCanvas("cnv","",1500,1000)
cnv.Divide(3,2)
cnv.cd(1)
ROOT.gPad.SetLogy()
ROOT.gPad.SetTicks(1,1)
histos["SMALL_hdx_vs_E_isGauss_N1"].Draw("colz")
ROOT.gPad.RedrawAxis()
cnv.cd(2)
ROOT.gPad.SetLogy()
ROOT.gPad.SetTicks(1,1)
histos["SMALL_hdx_vs_E_isGauss_N3"].Draw("colz")
ROOT.gPad.RedrawAxis()
cnv.cd(3)
ROOT.gPad.SetLogy()
ROOT.gPad.SetTicks(1,1)
histos["SMALL_hdx_vs_E_isGauss_N0"].Draw("colz")
ROOT.gPad.RedrawAxis()
cnv.cd(4)
ROOT.gPad.SetLogy()
ROOT.gPad.SetLogz()
ROOT.gPad.SetTicks(1,1)
histos["SMALL_hdx_vs_E_N1"].Draw("colz")
ROOT.gPad.RedrawAxis()
cnv.cd(5)
ROOT.gPad.SetLogy()
ROOT.gPad.SetLogz()
ROOT.gPad.SetTicks(1,1)
histos["SMALL_hdx_vs_E_N3"].Draw("colz")
ROOT.gPad.RedrawAxis()
cnv.cd(6)
ROOT.gPad.SetLogy()
ROOT.gPad.SetLogz()
ROOT.gPad.SetTicks(1,1)
histos["SMALL_hdx_vs_E_N0"].Draw("colz")
ROOT.gPad.RedrawAxis()
cnv.SaveAs(pdf)


cnv = TCanvas("cnv","",500,500)
cnv.Divide(2,2)
cnv.cd(1)
ROOT.gPad.SetTicks(1,1)
test0.Draw("hist")
ROOT.gPad.RedrawAxis()
cnv.cd(2)
ROOT.gPad.SetTicks(1,1)
test1.Draw("hist")
ROOT.gPad.RedrawAxis()
cnv.cd(3)
ROOT.gPad.SetTicks(1,1)
test2.Draw("hist")
ROOT.gPad.RedrawAxis()
cnv.cd(4)
ROOT.gPad.SetTicks(1,1)
test3.Draw("hist")
ROOT.gPad.RedrawAxis()
cnv.SaveAs(pdf+")")

print("Writing root file...")
tfout = TFile("out.root","RECREATE")
tfout.cd()
for name,h in histos.items(): h.Write()
tfout.Write()
tfout.Close()
