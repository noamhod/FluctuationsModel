import math
import array
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, rfft, irfft
from scipy.special import sici, exp1
from scipy.signal import convolve, fftconvolve
from scipy.interpolate import interp1d
import ROOT

import constants as C
import units as U
import material as mat
import particle as prt
import bins
import fluctuations as flct
import hist
import model

import argparse
parser = argparse.ArgumentParser(description='scan_prostprocess.py...')
parser.add_argument('-G', metavar='G=1 will plot the gifs, G=0 will skip this',    required=True,  help='G=1 will plot the gifs, G=0 will skip this')
argus = parser.parse_args()
G = int(argus.G)
dogif = (G==1)

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.15)

ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning

rootpath = "/Users/noamtalhod/tmp/root"
pngpath  = "/Users/noamtalhod/tmp/png" 

#########################
### make gif for all bins
print("\nClean temp png's and temp png path...")
ROOT.gSystem.Unlink("scan_pdfs.gif") ## remove old files
ROOT.gSystem.Unlink("scan_cdfs.gif") ## remove old files
ROOT.gSystem.Exec("/bin/rm -f scan_pdfs.gif scan_cdfs.gif") ## remove old files
ROOT.gSystem.Exec(f"/bin/rm -rf {pngpath}") ## remove old files
ROOT.gSystem.Exec(f"/bin/mkdir -p {pngpath}")


def plot_slices(label,build,E,L,hists,pdffile):
    NrawSteps = hists["hE_"+label].GetEntries()
    cnt_pdf = hists[label+"_cnt_pdf"] if(type(hists[label+"_cnt_pdf"]) is ROOT.TH1D) else None
    sec_pdf = hists[label+"_sec_pdf"] if(type(hists[label+"_sec_pdf"]) is ROOT.TH1D) else None
    cnt_cdf = hists[label+"_cnt_cdf"] if(type(hists[label+"_cnt_cdf"]) is ROOT.TH1D) else None
    sec_cdf = hists[label+"_sec_cdf"] if(type(hists[label+"_sec_cdf"]) is ROOT.TH1D) else None
    
    cnt_slice_cdf = hists["hdEcnt_"+label].Clone("hcnt_cdf_"+label) if(hists["hdEcnt_"+label] is not None) else None
    sec_slice_cdf = hists["hdEsec_"+label].Clone("hsec_cdf_"+label) if(hists["hdEsec_"+label] is not None) else None
    
    if(cnt_slice_cdf is not None and cnt_slice_cdf.Integral()>0):
        if(cnt_slice_cdf.Integral()>0): cnt_slice_cdf.Scale(1./cnt_slice_cdf.Integral())
        cnt_slice_cdf = cnt_slice_cdf.GetCumulative()
    if(sec_slice_cdf is not None and sec_slice_cdf.Integral()>0):
        if(sec_slice_cdf.Integral()>0): sec_slice_cdf.Scale(1./sec_slice_cdf.Integral())
        sec_slice_cdf = sec_slice_cdf.GetCumulative()
    
    ##########################
    cgif_pdfs = ROOT.TCanvas("pdf_"+label,"",1000,1000)
    cgif_pdfs.Divide(2,2)
    cgif_pdfs.cd(1)
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    hists["hdEcnt_"+label].Draw("hist")
    if(cnt_pdf is not None):
        hmax = hist.find_h_max(cnt_pdf)
        if(hmax>0): cnt_pdf.Scale(hists["hdEcnt_"+label].GetMaximum() / hmax)
        cnt_pdf.SetLineWidth(2)
        cnt_pdf.Draw("hist same")
    s = ROOT.TLatex() ### the text
    s.SetNDC(1);
    s.SetTextAlign(13);
    s.SetTextFont(22);
    s.SetTextColor(ROOT.kBlack)
    s.SetTextSize(0.04)
    modtitle = build.replace("->"," #otimes ").replace(".","")
    modtitle = modtitle.replace(" #otimes SECB","")
    s.DrawLatex(0.18,0.25,modtitle)
    ROOT.gPad.RedrawAxis()
    ##########################
    cgif_pdfs.cd(2)
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    hists["hE_"+label].Draw("hist")
    s = ROOT.TLatex() ### the text
    s.SetNDC(1);
    s.SetTextAlign(13);
    s.SetTextFont(22);
    s.SetTextColor(ROOT.kBlack)
    s.SetTextSize(0.04)
    s.DrawLatex(0.15,0.86,ROOT.Form("E=%.3e #in [%.3e, %.3e) [MeV]" % (E,hists["hE_"+label].GetXaxis().GetXmin(), hists["hE_"+label].GetXaxis().GetXmax())))
    s.DrawLatex(0.15,0.81,ROOT.Form("N raw steps = %d" % (NrawSteps)))
    ROOT.gPad.RedrawAxis()
    ROOT.gPad.Update()
    ##########################
    cgif_pdfs.cd(3)
    if(hists["hdEsec_"+label].GetXaxis().GetXmin()>0): ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    hists["hdEsec_"+label].Draw("hist")
    if(sec_pdf is not None):
        hmax = hist.find_h_max(sec_pdf)
        if(hmax>0): sec_pdf.Scale(hists["hdEsec_"+label].GetMaximum() / hmax)
        sec_pdf.Draw("hist same")
        s = ROOT.TLatex() ### the text
        s.SetNDC(1);
        s.SetTextAlign(13);
        s.SetTextFont(22);
        s.SetTextColor(ROOT.kBlack)
        s.SetTextSize(0.04)
        modtitle = "SECB"
        s.DrawLatex(0.18,0.25,modtitle)
    ROOT.gPad.RedrawAxis()
    ROOT.gPad.Update()
    ##########################
    cgif_pdfs.cd(4)
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    hists["hdL_"+label].Draw("hist")
    s = ROOT.TLatex() ### the text
    s.SetNDC(1);
    s.SetTextAlign(13);
    s.SetTextFont(22);
    s.SetTextColor(ROOT.kBlack)
    s.SetTextSize(0.04)
    s.DrawLatex(0.15,0.86,ROOT.Form("#DeltaL=%.3e #in [%.3e, %.3e) [#mum]" % (L,hists["hdL_"+label].GetXaxis().GetXmin(), hists["hdL_"+label].GetXaxis().GetXmax())))
    s.DrawLatex(0.15,0.81,ROOT.Form("N raw steps = %d" % (NrawSteps)))
    ROOT.gPad.RedrawAxis()
    ROOT.gPad.Update()
    ##########################
    cgif_pdfs.Update()
    cgif_pdfs.Print(f"{pngpath}/scan_pdfs_{label}.png")

    ##########################
    cgif_cdfs = ROOT.TCanvas("cdf_"+label,"",1000,1000)
    cgif_cdfs.Divide(2,2)
    cgif_cdfs.cd(1)
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    if(cnt_slice_cdf is not None and cnt_slice_cdf.Integral()>0):
        cnt_slice_cdf.SetMinimum(1.e-5)
        cnt_slice_cdf.SetMaximum(2.e0)
        cnt_slice_cdf.Draw("hist")
    if(cnt_cdf is not None): cnt_cdf.Draw("hist same")
    s = ROOT.TLatex() ### the text
    s.SetNDC(1);
    s.SetTextAlign(13);
    s.SetTextFont(22);
    s.SetTextColor(ROOT.kBlack)
    s.SetTextSize(0.04)
    modtitle = build.replace("->"," #otimes ").replace(".","")
    modtitle = modtitle.replace(" #otimes SECB","")
    s.DrawLatex(0.18,0.25,modtitle)
    ROOT.gPad.RedrawAxis()
    ##########################
    cgif_cdfs.cd(2)
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    hists["hE_"+label].Draw("hist")
    s = ROOT.TLatex() ### the text
    s.SetNDC(1);
    s.SetTextAlign(13);
    s.SetTextFont(22);
    s.SetTextColor(ROOT.kBlack)
    s.SetTextSize(0.04)
    s.DrawLatex(0.15,0.86,ROOT.Form("E=%.3e #in [%.3e, %.3e) [MeV]" % (E*U.eV2MeV,hists["hE_"+label].GetXaxis().GetXmin(), hists["hE_"+label].GetXaxis().GetXmax())))
    s.DrawLatex(0.15,0.81,ROOT.Form("N raw steps = %d" % (NrawSteps)))
    ROOT.gPad.RedrawAxis()
    ROOT.gPad.Update()
    ##########################
    cgif_cdfs.cd(3)
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    if(sec_slice_cdf is not None and sec_slice_cdf.Integral()>0):
        sec_slice_cdf.SetMinimum(1.e-5)
        sec_slice_cdf.SetMaximum(2.e0)
        sec_slice_cdf.Draw("hist")
        s = ROOT.TLatex() ### the text
        s.SetNDC(1);
        s.SetTextAlign(13);
        s.SetTextFont(22);
        s.SetTextColor(ROOT.kBlack)
        s.SetTextSize(0.04)
        modtitle = "SECB"
        s.DrawLatex(0.18,0.25,modtitle)
    if(sec_cdf is not None): sec_cdf.Draw("hist same")
    ROOT.gPad.RedrawAxis()
    ROOT.gPad.Update()
    ##########################
    cgif_cdfs.cd(4)
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    hists["hdL_"+label].Draw("hist")
    s = ROOT.TLatex() ### the text
    s.SetNDC(1);
    s.SetTextAlign(13);
    s.SetTextFont(22);
    s.SetTextColor(ROOT.kBlack)
    s.SetTextSize(0.04)
    s.DrawLatex(0.15,0.86,ROOT.Form("#DeltaL=%.3e #in [%.3e, %.3e) [#mum]" % (L*U.cm2um,hists["hdL_"+label].GetXaxis().GetXmin(), hists["hdL_"+label].GetXaxis().GetXmax())))
    s.DrawLatex(0.15,0.81,ROOT.Form("N raw steps = %d" % (NrawSteps)))
    ROOT.gPad.RedrawAxis()
    ROOT.gPad.Update()
    ##########################
    cgif_cdfs.Update()
    cgif_cdfs.Print(f"{pngpath}/scan_cdfs_{label}.png")
    
    print(f"Finished plotting slice: {label}")



tf0 = ROOT.TFile("scan_example.root","READ")
href = tf0.Get("SMALL_h_dL_vs_E")

hkst_prob = href.Clone("KSprob")
hkst_dist = href.Clone("KSdist")
hc2t_ndof = href.Clone("C2ndof")
hkst_prob.Reset()
hkst_dist.Reset()
hc2t_ndof.Reset()
hkst_prob.GetZaxis().SetTitle("KS test probability")
hkst_dist.GetZaxis().SetTitle("KS test distance")
hc2t_ndof.GetZaxis().SetTitle("#chi^{2}/N_{DoF} test")

pdffile = "scan_slices.pdf"
cnv = ROOT.TCanvas("cnv", "", 1000,1000)
cnv.SaveAs(pdffile+"(")

nall = href.GetNbinsX()*href.GetNbinsY()
nslices = 0
for ie in range(1,href.GetNbinsX()+1):
    for il in range(1,href.GetNbinsY()+1):
        
        ### skip empty slices (no GEANT4 data)
        if(href.GetBinContent(ie,il)<1): continue
        
        ### get the slice details
        label = "E"+str(ie)+"_dL"+str(il)
        E     = href.GetXaxis().GetBinCenter(ie) ## already in MeV
        L     = href.GetYaxis().GetBinCenter(il) ## already in um

        ### get the rootfile
        tf = ROOT.TFile.Open(f"{rootpath}/rootslice_{label}.root","READ")
        build = tf.Get("build").GetTitle()

        ### get the histos from the file
        hists = {
           "hdEcnt_"+label  : tf.Get("hdEcnt_"+label),
           "hdEsec_"+label  : tf.Get("hdEsec_"+label),
           "hE_"+label      : tf.Get("hE_"+label),
           "hdL_"+label     : tf.Get("hdL_"+label),
           label+"_cnt_pdf" : tf.Get(label+"_cnt_pdf"),
           label+"_sec_pdf" : tf.Get(label+"_sec_pdf"),
           label+"_cnt_cdf" : tf.Get(label+"_cnt_cdf"),
           label+"_sec_cdf" : tf.Get(label+"_sec_cdf"),
        }
        
        ### for the summary
        tree = tf.Get("meta_"+label)
        for evt in tree:
            KS_prob = evt.KS_prob_test_cnt_pdf
            KS_dist = evt.KS_dist_test_cnt_pdf if(evt.KS_dist_test_cnt_pdf>0) else 1e-10
            C2_ndof = evt.C2_ndof_test_cnt_pdf if(evt.C2_ndof_test_cnt_pdf>0) else 1e-10
            hkst_prob.SetBinContent(ie,il,KS_prob)
            hkst_dist.SetBinContent(ie,il,KS_dist)
            hc2t_ndof.SetBinContent(ie,il,C2_ndof)
        
        ### plot the slice
        if(dogif): plot_slices(label,build,E,L,hists,pdffile)
        
        ### close the file
        tf.Close()
        
        if(nslices%100==0 and nslices!=0): print(f"Processed {nslices} slices out of {nall} total")
        nslices += 1

print(f"\nProcessed {nslices} slices with at least 1 step, out of {nall} total")


cnv = ROOT.TCanvas("cnv", "", 1000,1000)
cnv.SaveAs(pdffile+")")

# gridx,gridy = hist.getGrid(href)
# for line in gridx: line.SetLineColor(ROOT.kGray)
# for line in gridy: line.SetLineColor(ROOT.kGray)

canvas = ROOT.TCanvas("canvas", "canvas", 1000,1000)
canvas.Divide(2,2)
canvas.cd(1)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogy()
# ROOT.gPad.SetLogz()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.18)
hkst_prob.GetZaxis().SetTitleOffset(1.6)
href.Draw("colz")
# for line in gridx: line.Draw("same")
# for line in gridy: line.Draw("same")
ROOT.gPad.RedrawAxis()
canvas.cd(2)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogy()
ROOT.gPad.SetLogz()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.18)
hkst_prob.GetZaxis().SetTitleOffset(1.6)
hkst_prob.Draw("colz")
# for line in gridx: line.Draw("same")
# for line in gridy: line.Draw("same")
ROOT.gPad.RedrawAxis()
canvas.cd(3)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogy()
# ROOT.gPad.SetLogz()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.18)
hkst_dist.GetZaxis().SetTitleOffset(1.6)
hkst_dist.Draw("colz")
# for line in gridx: line.Draw("same")
# for line in gridy: line.Draw("same")
ROOT.gPad.RedrawAxis()
canvas.cd(4)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogy()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.18)
hc2t_ndof.GetZaxis().SetTitleOffset(1.5)
hc2t_ndof.Draw("colz")
# for line in gridx: line.Draw("same")
# for line in gridy: line.Draw("same")
ROOT.gPad.RedrawAxis()
canvas.SaveAs("test_kschisq.pdf")


#####################
### finalize the gifs
if(dogif):
    print("\nMaking gif for pdfs...")
    ROOT.gSystem.Exec(f"magick -delay 0.01 $(ls {pngpath}/scan_pdfs_*.png | sort -V) scan_pdfs.gif")
    print("\nMaking gif for cdfs...")
    ROOT.gSystem.Exec(f"magick -delay 0.01 $(ls {pngpath}/scan_cdfs_*.png | sort -V) scan_cdfs.gif")
