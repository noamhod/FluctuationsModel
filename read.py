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


### first normalize
hmin_dE, hmax_dE      = hist.hNorm(histos,"SMALL_hdxinv_vs_E","E","dxinv","dE")
hmin_E,hmax_E         = hist.hNorm(histos,"SMALL_hdxinv_vs_E","E","dxinv","E")
hmin_dxinv,hmax_dxinv = hist.hNorm(histos,"SMALL_hdxinv_vs_E","E","dxinv","dxinv")
hmin_dx,hmax_dx       = hist.hNorm(histos,"SMALL_hdxinv_vs_E","E","dx","dx")


#############################################################
#############################################################
#############################################################
TargetMat = mat.Si # or e.g. mat.Al
ParticleN = "Proton"
ParticleM = C.mp
ParticleQ = +1
ParamName = ParticleN+"_on_"+TargetMat.name
dEdxModel = "G4:Tcut" # or "BB:Tcut"
par       = flct.Parameters(ParamName,ParticleM,ParticleQ,TargetMat,dEdxModel,"inputs/eloss_p_si.txt","inputs/BB.csv")
#############################################################
#############################################################
#############################################################



test0 = ROOT.TH1D("test0",";Theory_{#sigma}/Hist_{#sigma}",100,0,2)
test1 = ROOT.TH1D("test1",";Theory_{MPV}/Hist_{MPV}",100,0,2)
test2 = ROOT.TH1D("test2",";Theory_{Mean}/Hist_{Mean}",100,0,2)
test3 = ROOT.TH1D("test3",";Theory_{MPV}/PDG_{MPV}",100,0,2)


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
        ### skip E-x bin if there are too few raw steps
        # if(NrawSteps<10): continue

        ### get the E and x before skipping ay E-X bin
        midRangeE = (histos["hE_"+label].GetXaxis().GetXmax()-histos["hE_"+label].GetXaxis().GetXmin())/2.
        midRangeX = (histos["hdx_"+label].GetXaxis().GetXmax()-histos["hdx_"+label].GetXaxis().GetXmin())/2.
        E = histos["hE_"+label].GetMean()*U.MeV2eV if(NrawSteps>=NminRawSteps) else midRangeE*U.MeV2eV # eV
        x = histos["hdx_"+label].GetMean()*U.um2cm if(NrawSteps>=NminRawSteps) else midRangeX*U.um2cm  # cm

        ######################################
        ### Build the model shapes
        modelpars = par.GetModelPars(E,x)
        Mod = model.Model(x,E,modelpars)
        Mod.set_all_shapes()
        cnt_pdfs_scaled = Mod.cnt_pdfs_scaled
        sec_pdfs        = Mod.sec_pdfs
        ######################################
        
        cgif = ROOT.TCanvas("gif","",1000,1000)
        cgif.Divide(2,2)
        cgif.cd(1)
        ROOT.gPad.SetLogx()
        ROOT.gPad.SetLogy()
        ROOT.gPad.SetTicks(1,1)
        histos[name].DrawNormalized("hist")
        sec_pdfs["hModel"].SetFillColorAlpha(sec_pdfs["hModel"].GetLineColor(),0.30)
        sec_pdfs["hModel"].DrawNormalized("hist same")
        histos[name].DrawNormalized("hist same") ## redraw the hist on top
        ROOT.gPad.RedrawAxis()
        
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
        s = ROOT.ROOT.TLatex() ### the text
        s.SetNDC(1);
        s.SetTextAlign(13);
        s.SetTextFont(22);
        s.SetTextColor(ROOT.kBlack)
        s.SetTextSize(0.04)
        s.DrawLatex(0.3,0.90,ROOT.Form("E #in [%.3e, %.3e) [MeV]" % (histos["hE_"+label].GetXaxis().GetXmin(), histos["hE_"+label].GetXaxis().GetXmax())))
        s.DrawLatex(0.3,0.84,ROOT.Form("dx #in [%.3e, %.3e) [#mum]" % (histos["hdx_"+label].GetXaxis().GetXmin(), histos["hdx_"+label].GetXaxis().GetXmax())))
        s.DrawLatex(0.3,0.78,ROOT.Form("N raw steps = %d" % (NrawSteps)))
        ROOT.gPad.RedrawAxis()
        ROOT.gPad.Update()
        
        cgif.cd(4)
        name = "hdx_"+label
        ROOT.gPad.SetLogy()
        ROOT.gPad.SetTicks(1,1)
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


print("Writing root file...")
tfout = ROOT.TFile("out.root","RECREATE")
tfout.cd()
for name,h in histos.items(): h.Write()
tfout.Write()
tfout.Close()
