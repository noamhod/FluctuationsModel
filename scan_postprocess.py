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
if(dogif):
    print("\nClean temp png's and temp png path...")
    ROOT.gSystem.Unlink("scan_cnt_pdfs.gif") ## remove old files
    ROOT.gSystem.Unlink("scan_cnt_cdfs.gif") ## remove old files
    ROOT.gSystem.Unlink("scan_sec_pdfs.gif") ## remove old files
    ROOT.gSystem.Unlink("scan_sec_cdfs.gif") ## remove old files
    ROOT.gSystem.Exec("/bin/rm -f scan_cnt_pdfs.gif scan_cnt_cdfs.gif") ## remove old files
    ROOT.gSystem.Exec("/bin/rm -f scan_sec_pdfs.gif scan_sec_cdfs.gif") ## remove old files
    ROOT.gSystem.Exec(f"/bin/rm -rf {pngpath}") ## remove old files
    ROOT.gSystem.Exec(f"/bin/mkdir -p {pngpath}")


def avg_cdf_distance(h1,h2):
    if(h1.GetNbinsX()!=h2.GetNbinsX()):
        print("Histogram bin numbers do not match. Quitting")
        quit()
    avg_dist = 0
    Nx = h1.GetNbinsX()
    for bx in range(1,Nx+1):
        y1 = h1.GetBinContent(bx)
        y2 = h2.GetBinContent(bx)
        avg_dist += abs(y1-y2)
    return avg_dist/float(Nx)

def rms_cdf_distance(h1,h2):
    if(h1.GetNbinsX()!=h2.GetNbinsX()):
        print("Histogram bin numbers do not match. Quitting")
        quit()
    rms_dist = 0
    Nx = h1.GetNbinsX()
    for bx in range(1,Nx+1):
        y1 = h1.GetBinContent(bx)
        y2 = h2.GetBinContent(bx)
        rms_dist += (y1-y2)**2
    return math.sqrt(rms_dist/float(Nx))

def plot_continuous_slices(label,build,E,L,hists):
    NrawSteps = hists["hE_"+label].GetEntries()
    cnt_pdf = hists[label+"_cnt_pdf"] if(type(hists[label+"_cnt_pdf"]) is ROOT.TH1D) else None
    cnt_cdf = hists[label+"_cnt_cdf"] if(type(hists[label+"_cnt_cdf"]) is ROOT.TH1D) else None
    cnt_slice_cdf = hists["hdEcnt_"+label].Clone("hcnt_cdf_"+label) if(hists["hdEcnt_"+label] is not None) else None
    if(cnt_slice_cdf is not None and cnt_slice_cdf.Integral()>0):
        if(cnt_slice_cdf.Integral()>0): cnt_slice_cdf.Scale(1./cnt_slice_cdf.Integral())
        cnt_slice_cdf = cnt_slice_cdf.GetCumulative()
    
    ##########################
    cgif_pdfs = ROOT.TCanvas("pdf_"+label,"",1500,500)
    cgif_pdfs.Divide(3,1)
    cgif_pdfs.cd(1)
    if(("BEBL" not in build) and hists["hdEcnt_"+label].Integral()>0): ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    hists["hdEcnt_"+label].Draw("hist")
    if(cnt_pdf is not None):
        hmax = hist.find_h_max(cnt_pdf)
        # hint = hist.get_h_int(cnt_pdf)
        hg4max = hists["hdEcnt_"+label].GetMaximum()
        # hg4int = hist.get_h_int(hists["hdEcnt_"+label])
        if(hmax>0 and hg4max>0): cnt_pdf.Scale(hg4max / hmax)
        # cnt_pdf.Scale(hg4int / hint)
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
    cgif_pdfs.Print(f"{pngpath}/scan_cnt_pdfs_{label}.png")

    ##########################
    cgif_cdfs = ROOT.TCanvas("cdf_"+label,"",1500,500)
    cgif_cdfs.Divide(3,1)
    cgif_cdfs.cd(1)
    if(("BEBL" not in build) and cnt_slice_cdf.Integral()>0): ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    if(cnt_slice_cdf is not None and cnt_slice_cdf.Integral()>0):
        cnt_slice_cdf.SetMinimum(1.e-5)
        cnt_slice_cdf.SetMaximum(2.e0)
        cnt_slice_cdf.Draw("hist")
        if(cnt_cdf is not None): cnt_cdf.Draw("hist same")
    else:
        if(cnt_cdf is not None):
            cnt_cdf.SetMinimum(1.e-5)
            cnt_cdf.SetMaximum(2.e0)
            cnt_cdf.Draw("hist")
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
    s.DrawLatex(0.15,0.86,ROOT.Form("E=%.3e #in [%.3e, %.3e) [MeV]" % (E,hists["hE_"+label].GetXaxis().GetXmin(), hists["hE_"+label].GetXaxis().GetXmax())))
    s.DrawLatex(0.15,0.81,ROOT.Form("N raw steps = %d" % (NrawSteps)))
    ROOT.gPad.RedrawAxis()
    ROOT.gPad.Update()
    ##########################
    cgif_cdfs.cd(3)
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
    cgif_cdfs.Update()
    cgif_cdfs.Print(f"{pngpath}/scan_cnt_cdfs_{label}.png")

    print(f"Finished plotting continuous slice: {label}")


def plot_secondaries_slices(label,build,E,hists):
    NrawSteps = hists["hE_"+label].GetEntries()
    sec_pdf = hists[label+"_sec_pdf"] if(type(hists[label+"_sec_pdf"]) is ROOT.TH1D) else None
    sec_cdf = hists[label+"_sec_cdf"] if(type(hists[label+"_sec_cdf"]) is ROOT.TH1D) else None
    sec_slice_cdf = hists["hdEsec_"+label].Clone("hsec_cdf_"+label) if(hists["hdEsec_"+label] is not None) else None
    if(sec_slice_cdf is not None and sec_slice_cdf.Integral()>0):
        if(sec_slice_cdf.Integral()>0): sec_slice_cdf.Scale(1./sec_slice_cdf.Integral())
        sec_slice_cdf = sec_slice_cdf.GetCumulative()

    ##########################
    cgif_pdfs = ROOT.TCanvas("pdf_"+label,"",1000,500)
    cgif_pdfs.Divide(2,1)
    ##########################
    cgif_pdfs.cd(1)
    if(hists["hdEsec_"+label].Integral()>0 and sec_pdf is not None and sec_pdf.Integral()>0): ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    hists["hdEsec_"+label].Draw("hist")
    if(sec_pdf is not None):
        hmax = hist.find_h_max(sec_pdf)
        # hint = hist.get_h_int(sec_pdf)
        hg4max = hists["hdEsec_"+label].GetMaximum()
        # hg4int = hist.get_h_int(hists["hdEsec_"+label])
        if(hmax>0 and hg4max>0): sec_pdf.Scale(hg4max / hmax)
        # sec_pdf.Scale(hg4int / hint)
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
    cgif_pdfs.Update()
    cgif_pdfs.Print(f"{pngpath}/scan_sec_pdfs_{label}.png")

    ##########################
    cgif_cdfs = ROOT.TCanvas("cdf_"+label,"",1000,500)
    cgif_cdfs.Divide(2,1)
    ##########################
    cgif_cdfs.cd(1)
    # if(sec_slice_cdf.Integral()>0 and sec_cdf is not None and sec_cdf.Integral()>0): ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogx()
    ROOT.gPad.SetLogy()
    ROOT.gPad.SetTicks(1,1)
    # if(sec_slice_cdf is not None and sec_slice_cdf.Integral()>0):
    if(sec_slice_cdf is not None):
        sec_slice_cdf.SetMinimum(1.e-5)
        sec_slice_cdf.SetMaximum(2.e0)
        sec_slice_cdf.Draw("hist")
        if(sec_cdf is not None): sec_cdf.Draw("hist same")
    else:
        if(sec_cdf is not None):
            sec_cdf.SetMinimum(1.e-5)
            sec_cdf.SetMaximum(2.e0)
            sec_cdf.Draw("hist")
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
    s.DrawLatex(0.15,0.86,ROOT.Form("E=%.3e #in [%.3e, %.3e) [MeV]" % (E,hists["hE_"+label].GetXaxis().GetXmin(), hists["hE_"+label].GetXaxis().GetXmax())))
    s.DrawLatex(0.15,0.81,ROOT.Form("N raw steps = %d" % (NrawSteps)))
    ROOT.gPad.RedrawAxis()
    ROOT.gPad.Update()
    ##########################
    cgif_cdfs.Update()
    cgif_cdfs.Print(f"{pngpath}/scan_sec_cdfs_{label}.png")
    
    print(f"Finished plotting secondaries slice: {label}")

################################################################
################################################################
################################################################


hE_slice_size = ROOT.TH1D("hE_slice_size",";E [MeV];Slice size in E [%]",              len(bins.Ebins_small)-1,array.array("d",bins.Ebins_small) )
hL_slice_size = ROOT.TH1D("hL_slice_size",";#DeltaL [#mum];Slice size in #DeltaL [%]", len(bins.dLbins_small)-1,array.array("d",bins.dLbins_small) )

tf0 = ROOT.TFile("scan_example.root","READ")
href_cnt = tf0.Get("SMALL_h_dL_vs_E")
href_sec = tf0.Get("SMALL_h_E")

hkst_cnt_prob = href_cnt.Clone("KSprob_cnt")
hkst_cnt_dist = href_cnt.Clone("KSdist_cnt")
hc2t_cnt_ndof = href_cnt.Clone("C2ndof_cnt")
havg_dist_cnt_ndof = href_cnt.Clone("AVGdist_cnt")
hrms_dist_cnt_ndof = href_cnt.Clone("RMSdist_cnt")
hkst_cnt_prob.Reset()
hkst_cnt_dist.Reset()
hc2t_cnt_ndof.Reset()
havg_dist_cnt_ndof.Reset()
hrms_dist_cnt_ndof.Reset()
hkst_cnt_prob.SetTitle("Continuous")
hkst_cnt_dist.SetTitle("Continuous")
hc2t_cnt_ndof.SetTitle("Continuous")
havg_dist_cnt_ndof.SetTitle("Continuous")
hrms_dist_cnt_ndof.SetTitle("Continuous")
hkst_cnt_prob.GetZaxis().SetTitle("KS test probability")
hkst_cnt_dist.GetZaxis().SetTitle("KS test max distance")
hc2t_cnt_ndof.GetZaxis().SetTitle("#chi^{2}/N_{DoF} test")
havg_dist_cnt_ndof.GetZaxis().SetTitle("Avg CDF distance test")
hrms_dist_cnt_ndof.GetZaxis().SetTitle("RMS CDF distance test")

hkst_sec_prob = href_sec.Clone("KSprob_sec")
hkst_sec_dist = href_sec.Clone("KSdist_sec")
hc2t_sec_ndof = href_sec.Clone("C2ndof_sec")
havg_dist_sec_ndof = href_sec.Clone("AVGdist_sec")
hrms_dist_sec_ndof = href_sec.Clone("RMSdist_sec")
hkst_sec_prob.Reset()
hkst_sec_dist.Reset()
hc2t_sec_ndof.Reset()
havg_dist_sec_ndof.Reset()
hrms_dist_sec_ndof.Reset()
hkst_sec_prob.SetTitle("Secondaries")
hkst_sec_dist.SetTitle("Secondaries")
hc2t_sec_ndof.SetTitle("Secondaries")
havg_dist_sec_ndof.SetTitle("Secondaries")
hrms_dist_sec_ndof.SetTitle("Secondaries")
hkst_sec_prob.GetYaxis().SetTitle("KS test probability")
hkst_sec_dist.GetYaxis().SetTitle("KS test max distance")
hc2t_sec_ndof.GetYaxis().SetTitle("#chi^{2}/N_{DoF} test")
havg_dist_sec_ndof.GetYaxis().SetTitle("Avg CDF distance test")
hrms_dist_sec_ndof.GetYaxis().SetTitle("RMS CDF distance test")





############################################################################
nall = href_cnt.GetNbinsX()*href_cnt.GetNbinsY()
nslices = 0
for ie in range(1,href_cnt.GetNbinsX()+1):
    for il in range(1,href_cnt.GetNbinsY()+1):
        
        ### get the slice details
        label = "E"+str(ie)+"_dL"+str(il)
        E     = href_cnt.GetXaxis().GetBinCenter(ie) ## already in MeV
        L     = href_cnt.GetYaxis().GetBinCenter(il) ## already in um
        DE    = href_cnt.GetXaxis().GetBinWidth(ie)
        DL    = href_cnt.GetYaxis().GetBinWidth(il)

        ### for plotting the relative slice size
        if(il==1): hE_slice_size.SetBinContent(ie,(DE/E)*100)
        if(ie==1): hL_slice_size.SetBinContent(il,(DL/L)*100)

        ### skip empty slices (no GEANT4 data)
        if(href_cnt.GetBinContent(ie,il)<1): continue

        ### get the rootfile
        tf = ROOT.TFile.Open(f"{rootpath}/slice_cnt_{label}.root","READ")
        build = tf.Get("build").GetTitle()

        ### get the histos from the file
        hists = {
           "hdEcnt_"+label  : tf.Get("hdEcnt_"+label),
           "hE_"+label      : tf.Get("hE_"+label),
           "hdL_"+label     : tf.Get("hdL_"+label),
           label+"_cnt_pdf" : tf.Get(label+"_cnt_pdf"),
           label+"_cnt_cdf" : tf.Get(label+"_cnt_cdf"),
        }
        
        epsilon = 1e-20

        # ### for the summary
        # tree = tf.Get("meta_"+label)
        # tree.GetEntry(0)
        #
        # if(L<1e-5): print(f"label={label}, E={E}, L={L}, KS_dist_cnt={tree.KS_dist_test_cnt_pdf}, C2_ndof_cnt={tree.C2_ndof_test_cnt_pdf}")
        #
        #
        # KS_prob_test_cnt = tree.KS_prob_test_cnt_pdf
        # KS_dist_test_cnt = tree.KS_dist_test_cnt_pdf
        # C2_ndof_test_cnt = tree.C2_ndof_test_cnt_pdf
        # if(KS_prob_test_cnt==0): KS_prob_test_cnt += epsilon
        # if(KS_dist_test_cnt==0): KS_dist_test_cnt += epsilon
        # if(C2_ndof_test_cnt==0): C2_ndof_test_cnt += epsilon
        # if(KS_prob_test_cnt>=0): hkst_cnt_prob.SetBinContent(ie,il, KS_prob_test_cnt)
        # if(KS_dist_test_cnt>=0): hkst_cnt_dist.SetBinContent(ie,il, KS_dist_test_cnt)
        # if(C2_ndof_test_cnt>=0): hc2t_cnt_ndof.SetBinContent(ie,il, C2_ndof_test_cnt)

        hcnt_pdf = hists["hdEcnt_"+label].Clone("hcnt_pdf")
        hcnt_pdf.Scale(1./hcnt_pdf.Integral())
        hcnt_cdf = hcnt_pdf.GetCumulative()

        KS_prob_test_cnt = hists["hdEcnt_"+label].KolmogorovTest(hists[label+"_cnt_pdf"])
        KS_dist_test_cnt = hists["hdEcnt_"+label].KolmogorovTest(hists[label+"_cnt_pdf"],"M")
        C2_ndof_test_cnt = hists["hdEcnt_"+label].Chi2Test(hists[label+"_cnt_pdf"],"CHI2/NDF")
        AVG_dist_test_cnt = avg_cdf_distance(hcnt_cdf,hists[label+"_cnt_cdf"])
        RMS_dist_test_cnt = rms_cdf_distance(hcnt_cdf,hists[label+"_cnt_cdf"])

        if(KS_prob_test_cnt==0): KS_prob_test_cnt += epsilon
        if(KS_dist_test_cnt==0): KS_dist_test_cnt += epsilon
        if(C2_ndof_test_cnt==0): C2_ndof_test_cnt += epsilon
        if(AVG_dist_test_cnt==0): AVG_dist_test_cnt += epsilon
        if(RMS_dist_test_cnt==0): RMS_dist_test_cnt += epsilon
        if(KS_prob_test_cnt>=0): hkst_cnt_prob.SetBinContent(ie,il, KS_prob_test_cnt)
        if(KS_dist_test_cnt>=0): hkst_cnt_dist.SetBinContent(ie,il, KS_dist_test_cnt)
        if(AVG_dist_test_cnt>=0): havg_dist_cnt_ndof.SetBinContent(ie,il, AVG_dist_test_cnt)
        if(RMS_dist_test_cnt>=0): hrms_dist_cnt_ndof.SetBinContent(ie,il, RMS_dist_test_cnt)

        del hcnt_pdf
        del hcnt_cdf
        
        ### plot the slice
        if(dogif): plot_continuous_slices(label,build,E,L,hists)
        
        ### close the file
        tf.Close()
        
        if(nslices%100==0 and nslices!=0): print(f"Processed {nslices} slices out of {nall} total")
        nslices += 1
print(f"\nProcessed {nslices} continuous slices with at least 1 step, out of {nall} total")
#########################################################################################



############################################################################
nall = href_sec.GetNbinsX()
nslices = 0
for ie in range(1,href_sec.GetNbinsX()+1):
    ### get the slice details
    label = "E"+str(ie)
    E     = href_cnt.GetXaxis().GetBinCenter(ie) ## already in MeV
    DE    = href_cnt.GetXaxis().GetBinWidth(ie)

    ### skip empty slices (no GEANT4 data)
    if(href_sec.GetBinContent(ie)<1): continue

    ### get the rootfile
    tf = ROOT.TFile.Open(f"{rootpath}/slice_sec_{label}.root","READ")
    build = tf.Get("build").GetTitle()

    ### get the histos from the file
    hists = {
       "hdEsec_"+label  : tf.Get("hdEsec_"+label),
       "hE_"+label      : tf.Get("hE_"+label),
       label+"_sec_pdf" : tf.Get(label+"_sec_pdf"),
       label+"_sec_cdf" : tf.Get(label+"_sec_cdf"),
    }
    epsilon = 1e-20

    # ### for the summary
    # tree = tf.Get("meta_"+label)
    # tree.GetEntry(0)
    #
    # if(L<1e-5): print(f"label={label}, E={E}, KS_dist_sec={tree.KS_dist_test_sec_pdf}, C2_ndof_sec={tree.C2_ndof_test_sec_pdf}")
    #
    #
    # KS_prob_test_sec = tree.KS_prob_test_sec_pdf
    # KS_dist_test_sec = tree.KS_dist_test_sec_pdf
    # C2_ndof_test_sec = tree.C2_ndof_test_sec_pdf
    # if(KS_prob_test_sec==0): KS_prob_test_sec += epsilon
    # if(KS_dist_test_sec==0): KS_dist_test_sec += epsilon
    # if(C2_ndof_test_sec==0): C2_ndof_test_sec += epsilon
    # if(KS_prob_test_sec>=0): hkst_sec_prob.SetBinContent(ie,il, KS_prob_test_sec)
    # if(KS_dist_test_sec>=0): hkst_sec_dist.SetBinContent(ie,il, KS_dist_test_sec)
    # if(C2_ndof_test_sec>=0): hc2t_sec_ndof.SetBinContent(ie,il, C2_ndof_test_sec)

    hsec_pdf = hists["hdEsec_"+label].Clone("hsec_pdf")
    hsec_pdf.Scale(1./hsec_pdf.Integral())
    hsec_cdf = hsec_pdf.GetCumulative()

    KS_prob_test_sec = hists["hdEsec_"+label].KolmogorovTest(hists[label+"_sec_pdf"])
    KS_dist_test_sec = hists["hdEsec_"+label].KolmogorovTest(hists[label+"_sec_pdf"],"M")
    C2_ndof_test_sec = hists["hdEsec_"+label].Chi2Test(hists[label+"_sec_pdf"],"CHI2/NDF")
    AVG_dist_test_sec = avg_cdf_distance(hsec_cdf,hists[label+"_sec_cdf"])
    RMS_dist_test_sec = rms_cdf_distance(hsec_cdf,hists[label+"_sec_cdf"])
    if(KS_prob_test_sec==0): KS_prob_test_sec += epsilon
    if(KS_dist_test_sec==0): KS_dist_test_sec += epsilon
    if(C2_ndof_test_sec==0): C2_ndof_test_sec += epsilon
    if(AVG_dist_test_sec==0): AVG_dist_test_sec += epsilon
    if(RMS_dist_test_sec==0): RMS_dist_test_sec += epsilon
    if(KS_prob_test_sec>=0): hkst_sec_prob.SetBinContent(ie, KS_prob_test_sec)
    if(KS_dist_test_sec>=0): hkst_sec_dist.SetBinContent(ie, KS_dist_test_sec)
    if(C2_ndof_test_sec>=0): hc2t_sec_ndof.SetBinContent(ie, C2_ndof_test_sec)
    if(AVG_dist_test_sec>=0): havg_dist_sec_ndof.SetBinContent(ie, AVG_dist_test_sec)
    if(RMS_dist_test_sec>=0): hrms_dist_sec_ndof.SetBinContent(ie, RMS_dist_test_sec)

    del hsec_pdf
    del hsec_cdf

    ### plot the slice
    if(dogif): plot_secondaries_slices(label,build,E,hists)

    ### close the file
    tf.Close()

    if(nslices%100==0 and nslices!=0): print(f"Processed {nslices} secondaries slices out of {nall} total")
    nslices += 1
print(f"\nProcessed {nslices} slices with at least 1 step, out of {nall} total")
#########################################################################################



# gridx,gridy = hist.getGrid(href_cnt)
# for line in gridx: line.SetLineColor(ROOT.kGray)
# for line in gridy: line.SetLineColor(ROOT.kGray)


canvas = ROOT.TCanvas("canvas", "canvas", 1200,500)
canvas.Divide(2,1)
canvas.cd(1)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogx()
# ROOT.gPad.SetLogy()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.18)
hE_slice_size.SetMinimum(0)
hE_slice_size.SetMaximum(50)
hE_slice_size.Draw("hist")
ROOT.gPad.RedrawAxis()
canvas.cd(2)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogx()
# ROOT.gPad.SetLogy()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.18)
hL_slice_size.SetMinimum(0)
hL_slice_size.SetMaximum(50)
hL_slice_size.Draw("hist")
ROOT.gPad.RedrawAxis()
canvas.SaveAs("test_kschisq.pdf(")




canvas = ROOT.TCanvas("canvas", "canvas", 1000,1000)
canvas.Divide(2,2)
canvas.cd(1)
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
canvas.cd(2)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.18)
# hkst_cnt_prob.GetZaxis().SetTitleOffset(1.6)
havg_dist_cnt_ndof.GetZaxis().SetTitleOffset(1.6)
# hkst_cnt_prob.Draw("colz")
havg_dist_cnt_ndof.Draw("colz")
# for line in gridx: line.Draw("same")
# for line in gridy: line.Draw("same")
ROOT.gPad.RedrawAxis()
canvas.cd(3)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.18)
hkst_cnt_dist.GetZaxis().SetTitleOffset(1.6)
hkst_cnt_dist.Draw("colz")
# for line in gridx: line.Draw("same")
# for line in gridy: line.Draw("same")
ROOT.gPad.RedrawAxis()
canvas.cd(4)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.18)
# hc2t_cnt_ndof.GetZaxis().SetTitleOffset(1.5)
hrms_dist_cnt_ndof.GetZaxis().SetTitleOffset(1.5)
# hc2t_cnt_ndof.Draw("colz")
hrms_dist_cnt_ndof.Draw("colz")
# for line in gridx: line.Draw("same")
# for line in gridy: line.Draw("same")
ROOT.gPad.RedrawAxis()
canvas.SaveAs("test_kschisq.pdf")



canvas = ROOT.TCanvas("canvas", "canvas", 1000,1000)
canvas.Divide(2,2)
canvas.cd(1)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.18)
href_sec.Draw("hist")
ROOT.gPad.RedrawAxis()
canvas.cd(2)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.18)
# hkst_sec_prob.GetZaxis().SetTitleOffset(1.6)
havg_dist_sec_ndof.GetZaxis().SetTitleOffset(1.6)
# hkst_sec_prob.Draw("hist")
havg_dist_sec_ndof.Draw("hist")
ROOT.gPad.RedrawAxis()
canvas.cd(3)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.18)
hkst_sec_dist.GetZaxis().SetTitleOffset(1.6)
hkst_sec_dist.Draw("hist")
ROOT.gPad.RedrawAxis()
canvas.cd(4)
ROOT.gPad.SetTicks(1,1)
ROOT.gPad.SetLogx()
ROOT.gPad.SetLogy()
ROOT.gPad.SetLeftMargin(0.15)
ROOT.gPad.SetRightMargin(0.18)
# hc2t_sec_ndof.GetZaxis().SetTitleOffset(1.5)
hrms_dist_sec_ndof.GetZaxis().SetTitleOffset(1.5)
# hc2t_sec_ndof.Draw("hist")
hrms_dist_sec_ndof.Draw("hist")
ROOT.gPad.RedrawAxis()
canvas.SaveAs("test_kschisq.pdf)")


fo = ROOT.TFile("test_kschisq.root","RECREATE")
fo.cd()
hkst_cnt_prob.Write()
hkst_cnt_dist.Write()
hc2t_cnt_ndof.Write()
hkst_sec_prob.Write()
hkst_sec_dist.Write()
hc2t_sec_ndof.Write()
fo.Write()
fo.Close()


# print("Split bins:")
# print(f"Nbins to split: {len(ybins2split)}")
# ybins2split.sort()
# print(ybins2split)

#####################
### finalize the gifs
if(dogif):
    print("\nMaking gif for continuous pdfs...")
    ROOT.gSystem.Exec(f"magick -delay 0.01 $(ls {pngpath}/scan_cnt_pdfs_*.png | sort -V) scan_cnt_pdfs.gif")
    print("\nMaking gif for continuous cdfs...")
    ROOT.gSystem.Exec(f"magick -delay 0.01 $(ls {pngpath}/scan_cnt_cdfs_*.png | sort -V) scan_cnt_cdfs.gif")

    print("\nMaking gif for secondaries pdfs...")
    ROOT.gSystem.Exec(f"magick -delay 0.01 $(ls {pngpath}/scan_sec_pdfs_*.png | sort -V) scan_sec_pdfs.gif")
    print("\nMaking gif for secondaries cdfs...")
    ROOT.gSystem.Exec(f"magick -delay 0.01 $(ls {pngpath}/scan_sec_cdfs_*.png | sort -V) scan_sec_cdfs.gif")

