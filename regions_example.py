import time
import pickle
import math
import array
import numpy as np
import pandas as pd
import ROOT
import units as U
import constants as C
import material as mat
import particle as prt
import bins
import fluctuations as flct
import hist
import model
import multiprocessing as mp


import argparse
parser = argparse.ArgumentParser(description='scan_example.py...')
parser.add_argument('-N', metavar='N steps to process if less than all, for all put 0', required=True,  help='N steps to process if les than all, for all put 0')
argus = parser.parse_args()
NN = int(argus.N)

ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.15)


if __name__ == "__main__":
    # open a file, where you stored the pickled data
    # fileY = open('data/with_secondaries/step_info_df_no_msc.pkl', 'rb')
    fileY = open('data/steps_info_18_07_2024.pkl', 'rb')
    # dump information to that file
    Y = pickle.load(fileY)
    # close the file
    fileY.close()
    df = pd.DataFrame(Y)
    # print(df)
    arr_dx     = df['dX'].to_numpy()
    arr_dy     = df['dY'].to_numpy()
    arr_dz     = df['dZ'].to_numpy()
    arr_dEcnt  = df['ContinuousLoss'].to_numpy()
    arr_dEtot  = df['TotalEnergyLoss'].to_numpy()
    arr_E      = df['KineticEnergy'].to_numpy()
    arr_dR     = df['dR'].to_numpy()
    arr_dL     = df['dL'].to_numpy()
    arr_dTheta = df['dTheta'].to_numpy()
    arr_dPhi   = df['dPhi'].to_numpy()


    ###################
    ### general histos:
    histos = {}
    hist.book(histos)
    
    ###################################
    ### get the parameters of the model
    dEdxModel  = "G4:Tcut" # or "BB:Tcut"
    TargetMat  = mat.Si # or e.g. mat.Al
    PrimaryPrt = prt.Particle(name="proton",meV=938.27208816*U.MeV2eV,mamu=1.007276466621,chrg=+1.,spin=0.5,lepn=0,magm=2.79284734463)
    par        = flct.Parameters(PrimaryPrt,TargetMat,dEdxModel,"inputs/dEdx_p_si.txt")
    

    #######################################
    ### Run the MC data and fill the histos
    print("\nStart the loop over GEANT4 data...")    
    for n in range(len(arr_dx)):
        dx     = arr_dx[n]*U.m2um
        dxinv  = 1/dx if(dx>0) else -999
        dy     = arr_dy[n]*U.m2um
        dz     = arr_dz[n]*U.m2um
        dEcnt  = arr_dEcnt[n]*U.eV2MeV
        dEtot  = arr_dEtot[n]*U.eV2MeV
        dEsec  = dEtot - dEcnt
        dE     = dEtot
        E      = arr_E[n]*U.eV2MeV
        dR     = arr_dR[n]*U.m2um
        dRinv = 1/dR if(dR>0) else -999 ## this happens for the primary particles...
        dL     = arr_dL[n]*U.m2um
        dTheta = arr_dTheta[n]
        dPhi   = arr_dPhi[n]
    
        ################
        ### valideations
        if(E>=bins.Emax):   continue ## skip the primary particles
        if(E<bins.Emin):    continue ## skip the low energy particles
        if(dx>=bins.dxmax): continue ## skip
        if(dx<bins.dxmin):  continue ## skip
        
        histos["hdL_vs_E"].Fill(E,dL)
        histos["SMALL_hdL_vs_E"].Fill(E,dL)
    
        if(n%1000000==0 and n>0): print("processed: ",n)
        if(n>NN and NN>0): break
    
    
    #################################################################################
    #################################################################################
    #################################################################################
    
    hRegions = {"BEBL":None,         "TGAU":None,          "IONBxEX1BxIONG":None, "IONBxIONGxEX1G":None,           "IONBxEX1B":None}
    hRegcols = {"BEBL":ROOT.kGray+2, "TGAU":ROOT.kGreen+3, "IONBxEX1B":ROOT.kRed, "IONBxEX1BxIONG":ROOT.kOrange+8, "IONBxIONGxEX1G":ROOT.kMagenta}
    Alphas   = {"BEBL":0.5,          "TGAU":0.5,           "IONBxEX1B":0.5,       "IONBxEX1BxIONG":0.5,            "IONBxIONGxEX1G":0.5}
    for name in hRegions:
        hRegions[name] = histos["SMALL_hdL_vs_E"].Clone(name)
        hRegions[name].Reset()
        hRegions[name].SetFillColorAlpha(hRegcols[name],Alphas[name])
    
    for bx in range(1,histos["SMALL_hdL_vs_E"].GetNbinsX()+1):
        for by in range(1,histos["SMALL_hdL_vs_E"].GetNbinsY()+1):
            E = histos["SMALL_hdL_vs_E"].GetXaxis().GetBinCenter(bx)*U.MeV2eV
            L = histos["SMALL_hdL_vs_E"].GetYaxis().GetBinCenter(by)*U.um2cm
            P = par.GetModelPars(E,L)
            Mod = model.Model(L,E,P)
            
            ### BEBL
            if(Mod.BEBL): hRegions["BEBL"].SetBinContent(bx,by,1)
            else:         hRegions["BEBL"].SetBinContent(bx,by,0)
            ### THK.GAUSS
            if(Mod.TGAU): hRegions["TGAU"].SetBinContent(bx,by,1)
            else:         hRegions["TGAU"].SetBinContent(bx,by,0)
            ### IONB and EX1B and IONG
            if(Mod.IONB and Mod.EX1B and Mod.IONG): hRegions["IONBxEX1BxIONG"].SetBinContent(bx,by,1)
            else:                                   hRegions["IONBxEX1BxIONG"].SetBinContent(bx,by,0)
            ### IONB and IONG and EX1G
            if(Mod.IONB and Mod.IONG and Mod.EX1G): hRegions["IONBxIONGxEX1G"].SetBinContent(bx,by,1)
            else:                                   hRegions["IONBxIONGxEX1G"].SetBinContent(bx,by,0)
            ### IONB and EX1B and not IONG and not EX1G
            if(Mod.IONB and Mod.EX1B and not Mod.IONG and not Mod.EX1G): hRegions["IONBxEX1B"].SetBinContent(bx,by,1)
            else:                                                        hRegions["IONBxEX1B"].SetBinContent(bx,by,0)
            
            if(not Mod.BEBL
               and not Mod.TGAU
               and not (Mod.IONB and Mod.EX1B and Mod.IONG)
               and not (Mod.IONB and Mod.IONG and Mod.EX1G)
               and not (Mod.IONB and Mod.EX1B and not Mod.IONG and not Mod.EX1G)):
                print(f"Model undefined for E={E*U.eV2MeV} MeV and L={L*U.cm2um} um --> build={Mod.build}")
    
    for name in hRegions:
        hRegions[name].Scale( histos["SMALL_hdL_vs_E"].GetMaximum() )
    
    
    

    ##########################################################
    pdf = "regions_example.pdf"
    #####################
    cnv = ROOT.TCanvas("cnv","",500,500)
    histos["hdL_vs_E"].SetTitle("BEBL")
    histos["hdL_vs_E"].Draw("colz")
    hRegions["BEBL"].Draw("box same")
    gridx,gridy = hist.getGrid(histos["SMALL_hdL_vs_E"])
    for line in gridx:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    for line in gridy:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.SaveAs(pdf+"(")
    #####################
    cnv = ROOT.TCanvas("cnv","",500,500)
    histos["hdL_vs_E"].SetTitle("IONB#otimesEX1B")
    histos["hdL_vs_E"].Draw("colz")
    hRegions["IONBxEX1B"].Draw("box same")
    gridx,gridy = hist.getGrid(histos["SMALL_hdL_vs_E"])
    for line in gridx:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    for line in gridy:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.SaveAs(pdf)
    #####################
    cnv = ROOT.TCanvas("cnv","",500,500)
    histos["hdL_vs_E"].SetTitle("IONB#otimesEX1B#otimesIONG")
    histos["hdL_vs_E"].Draw("colz")
    hRegions["IONBxEX1BxIONG"].Draw("box same")
    gridx,gridy = hist.getGrid(histos["SMALL_hdL_vs_E"])
    for line in gridx:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    for line in gridy:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.SaveAs(pdf)
    #####################
    cnv = ROOT.TCanvas("cnv","",500,500)
    histos["hdL_vs_E"].SetTitle("IONB#otimesIONG#otimesEX1G")
    histos["hdL_vs_E"].Draw("colz")
    hRegions["IONBxIONGxEX1G"].Draw("box same")
    gridx,gridy = hist.getGrid(histos["SMALL_hdL_vs_E"])
    for line in gridx:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    for line in gridy:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.SaveAs(pdf)
    #####################
    cnv = ROOT.TCanvas("cnv","",500,500)
    histos["hdL_vs_E"].SetTitle("THKG")
    histos["hdL_vs_E"].Draw("colz")
    hRegions["TGAU"].Draw("box same")
    gridx,gridy = hist.getGrid(histos["SMALL_hdL_vs_E"])
    for line in gridx:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    for line in gridy:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.SaveAs(pdf)
    #####################
    cnv = ROOT.TCanvas("cnv","",500,500)
    histos["hdL_vs_E"].SetTitle("")
    histos["hdL_vs_E"].Draw("colz")
    hRegions["BEBL"].Draw("box same")
    hRegions["TGAU"].Draw("box same")
    hRegions["IONBxEX1BxIONG"].Draw("box same")
    hRegions["IONBxIONGxEX1G"].Draw("box same")
    hRegions["IONBxEX1B"].Draw("box same")
    gridx,gridy = hist.getGrid(histos["SMALL_hdL_vs_E"])
    for line in gridx:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    for line in gridy:
        line.SetLineColor(ROOT.kGray)
        line.Draw("same")
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetLogz()
    ROOT.gPad.SetTicks(1,1)
    ROOT.gPad.RedrawAxis()
    cnv.SaveAs(pdf+")")

tfo = ROOT.TFile(pdf.replace(".pdf",".root"),"RECREATE")
tfo.cd()
for name,hreg in hRegions.items(): hreg.Write()
histos["hdL_vs_E"].Write()
tfo.Write()
tfo.Close()