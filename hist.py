import math
import array
import numpy as np
import ROOT
from ROOT import TH1D, TH2D, TLine
import bins

ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning

ROOT.gROOT.SetBatch(1)
ROOT.gStyle.SetOptFit(0)
ROOT.gStyle.SetOptStat(0)
ROOT.gStyle.SetPadBottomMargin(0.15)
ROOT.gStyle.SetPadLeftMargin(0.13)
ROOT.gStyle.SetPadRightMargin(0.15)


def getGrid(h):
    gridx = []
    gridy = []
    for e in range(1,h.GetNbinsX()+1):
        X    = h.GetXaxis().GetBinLowEdge(e)
        Ymin = h.GetYaxis().GetXmin()
        Ymax = h.GetYaxis().GetXmax()
        gridx.append( TLine(X,Ymin,X,Ymax) )
    for x in range(1,h.GetNbinsY()+1):
        Y    = h.GetYaxis().GetBinLowEdge(x)
        Xmin = h.GetXaxis().GetXmin()
        Xmax = h.GetXaxis().GetXmax()
        gridy.append( TLine(Xmin,Y,Xmax,Y) )
    return gridx,gridy

def getH1minmax(h):
    hmin = 1e20
    hmax = -1e20
    for b in range(1,h.GetNbinsX()+1):
        y = h.GetBinContent(b)
        if(y<=0): continue
        hmin = y if(y<hmin) else hmin
        hmax = y if(y>hmax) else hmax
    return hmin,hmax

def hNorm(hdict,href,first,second,var):
    hmin = 1e20
    hmax = 0
    for i1 in range(1,hdict[href].GetNbinsX()+1):
        for i2 in range(1,hdict[href].GetNbinsY()+1):
            name = f"h{var}_{first}{i1}_{second}{i2}"
            if(hdict[name].Integral()==0): continue
            hdict[name].Scale(1./hdict[name].Integral(), "width")
            hdict[name].GetYaxis().SetTitle( hdict[name].GetYaxis().GetTitle()+" per bin, Normalized" )
            ymin,ymax = getH1minmax(hdict[name])
            hmax = ymax if(ymax>hmax) else hmax
            hmin = ymin if(ymin<hmin) else hmin
    return hmin*0.5,hmax*2

def getFWHM(h):
    bmax = h.GetMaximumBin()
    xmax = h.GetBinCenter( bmax )
    hmax = hh.GetBinContent( bmax )
    bR = bmax
    y = hmax
    while(y>=hmax/2.):
        y = h.GetBinContent(bR)
        bR += 1
    
    bL = bmax
    y = hmax
    while(y>=hmax/2.):
        y = h.GetBinContent(bL)
        bL -= 1
    
    xL = h.GetXaxis().GetBinLowEdge(bL)
    xR = h.GetXaxis().GetBinUpEdge(bR)

    FWHM = xR-xL
    return FWHM


def getAvgY(h,isLogx=False,xbins=[]):
    ### this function assumes the x axis has linear binning!
    name  = h.GetName()+"_avgY"
    title = ";"+h.GetXaxis().GetTitle()+";"
    nbins = h.GetNbinsX()
    xmin  = h.GetXaxis().GetXmin()
    xmax  = h.GetXaxis().GetXmax()
    hAv   = TH1D(name,title,nbins,xmin,xmax) if(not isLogx) else TH1D(name,title,len(xbins)-1,xbins)
    for bx in range(1,h.GetNbinsX()+1):
        av = 0
        ev = 0
        for by in range(1,h.GetNbinsY()+1):
            n = h.GetBinContent(bx,by)
            y = h.GetYaxis().GetBinCenter(by)
            av += n*y
            ev += n
        av = av/ev if(ev!=0) else 0
        # if(isLogx): av = av/h.GetXaxis().GetBinWidth(bx)
        hAv.SetBinContent(bx,av)
    return hAv



def book(histos,emin=-1): ### must pass by reference!
    if(emin>0): bins.Emin = emin
    histos.update({"hE":           TH1D("h_E",";E [MeV];Steps", 1000,bins.Emin,bins.Emax)})
    histos.update({"hdE":          TH1D("h_dE",";#DeltaE [MeV];Steps", len(bins.dEbins)-1,array.array("d",bins.dEbins))}) # 1000,0,0.3)
    histos.update({"hdE_cnt":      TH1D("h_dE_cnt",";#DeltaE [MeV];Steps", len(bins.dEbins)-1,array.array("d",bins.dEbins))}) # 1000,0,0.3)
    histos.update({"hdE_sec":      TH1D("h_dE_sec",";#DeltaE [MeV];Steps", len(bins.dEbins)-1,array.array("d",bins.dEbins))}) # 1000,0,0.3)
    histos.update({"hdx":          TH1D("h_dx",";dx [#mum];Steps", len(bins.dxbins)-1,array.array("d",bins.dxbins))}) # 1000,0,+2e2)
    histos.update({"hdxinv":       TH1D("h_dxinv",";1/dx [1/#mum];Steps", len(bins.dxinvbins)-1,array.array("d",bins.dxinvbins))})
    histos.update({"hdR":          TH1D("h_dR",";dR [#mum];Steps", len(bins.dRbins)-1,array.array("d",bins.dRbins))}) # 1000,0,+5e-2)
    histos.update({"hdRinv":       TH1D("h_dRinv",";1/dR [1/#mum];Steps", len(bins.dRinvbins)-1,array.array("d",bins.dRinvbins))}) # 1000,0,+5e-2)
    histos.update({"hdEdx" :       TH1D("h_dEdx",";dE/dx [MeV/#mum];Steps", len(bins.dEdxbins)-1,array.array("d",bins.dEdxbins))})
    histos.update({"hdEdx_cnt" :   TH1D("h_dEdx_cnt",";dE/dx [MeV/#mum];Steps", len(bins.dEdxbins)-1,array.array("d",bins.dEdxbins))})
    histos.update({"hdEdx_sec" :   TH1D("h_dEdx_sec",";dE/dx [MeV/#mum];Steps", len(bins.dEdxbins)-1,array.array("d",bins.dEdxbins))})
    histos.update({"hdEdx_vs_E":   TH2D("h_dEdx_vs_E",";E [MeV];dE/dx [MeV/#mum];Steps",500,bins.Emin,bins.Emax, len(bins.dEdxbins)-1,array.array("d",bins.dEdxbins))})
    histos.update({"hdEdx_vs_E_small":     TH2D("h_dEdx_vs_E_small",";E [MeV];dE/dx [MeV/#mum];Steps",len(bins.Ebins_small)-1,bins.Ebins_small, len(bins.dEdxbins)-1,array.array("d",bins.dEdxbins))})
    histos.update({"hdEdx_vs_E_small_cnt": TH2D("h_dEdx_vs_E_small_cnt",";E [MeV];dE/dx [MeV/#mum];Steps",len(bins.Ebins_small)-1,bins.Ebins_small, len(bins.dEdxbins)-1,array.array("d",bins.dEdxbins))})
    histos.update({"hdEdx_vs_E_small_sec": TH2D("h_dEdx_vs_E_small_sec",";E [MeV];dE/dx [MeV/#mum];Steps",len(bins.Ebins_small)-1,bins.Ebins_small, len(bins.dEdxbins)-1,array.array("d",bins.dEdxbins))})
    histos.update({"hdE_vs_dx":    TH2D("h_dE_vs_dx",";dx [#mum];#DeltaE [MeV];Steps", len(bins.dxbins)-1,array.array("d",bins.dxbins), len(bins.dEbins)-1,array.array("d",bins.dEbins))})
    histos.update({"hdE_vs_dxinv": TH2D("h_dE_vs_dxinv",";1/dx [1/#mum];#DeltaE [MeV];Steps", len(bins.dxinvbins)-1,array.array("d",bins.dxinvbins), len(bins.dEbins)-1,array.array("d",bins.dEbins))})
    histos.update({"hdx_vs_E":     TH2D("h_dx_vs_E",";E [MeV];dx [#mum];Steps", 500,bins.Emin,bins.Emax, len(bins.dxbins)-1,array.array("d",bins.dxbins))})
    histos.update({"hdxinv_vs_E":  TH2D("h_dxinv_vs_E",";E [MeV];1/dx [1/#mum];Steps", 500,bins.Emin,bins.Emax, len(bins.dxinvbins)-1,array.array("d",bins.dxinvbins))})
    
    histos.update({"SMALL_hdx_vs_E_N1" : TH2D("SMALL_hdx_vs_E_N1",";E [MeV];dx [#mum];<n_{1}>", bins.n_small_E,bins.Emin,bins.Emax, len(bins.dxbins_small)-1,array.array("d",bins.dxbins_small))})
    histos.update({"SMALL_hdx_vs_E_N3" : TH2D("SMALL_hdx_vs_E_N3",";E [MeV];dx [#mum];<n_{3}>", bins.n_small_E,bins.Emin,bins.Emax, len(bins.dxbins_small)-1,array.array("d",bins.dxbins_small))})
    histos.update({"SMALL_hdx_vs_E_N0" : TH2D("SMALL_hdx_vs_E_N0",";E [MeV];dx [#mum];<n_{0}>", bins.n_small_E,bins.Emin,bins.Emax, len(bins.dxbins_small)-1,array.array("d",bins.dxbins_small))})
    
    histos.update({"SMALL_hdx_vs_E_isGauss_N1" : TH2D("SMALL_hdx_vs_E_isGauss_N1",";E [MeV];dx [#mum];Gaussian for <n_{1}>", bins.n_small_E,bins.Emin,bins.Emax, len(bins.dxbins_small)-1,array.array("d",bins.dxbins_small))})
    histos.update({"SMALL_hdx_vs_E_isGauss_N3" : TH2D("SMALL_hdx_vs_E_isGauss_N3",";E [MeV];dx [#mum];Gaussian for <n_{3}>", bins.n_small_E,bins.Emin,bins.Emax, len(bins.dxbins_small)-1,array.array("d",bins.dxbins_small))})
    histos.update({"SMALL_hdx_vs_E_isGauss_N0" : TH2D("SMALL_hdx_vs_E_isGauss_N0",";E [MeV];dx [#mum];Gaussian for <n_{0}>", bins.n_small_E,bins.Emin,bins.Emax, len(bins.dxbins_small)-1,array.array("d",bins.dxbins_small))})
    
    histos.update({"SMALL_hdx_vs_E" : TH2D("SMALL_h_dx_vs_E",";E [MeV];dx [#mum];Steps", bins.n_small_E,bins.Emin,bins.Emax, len(bins.dxbins_small)-1,array.array("d",bins.dxbins_small))})
    for ie in range(1,histos["SMALL_hdx_vs_E"].GetNbinsX()+1):
        label_E = str(ie)
        Emin = histos["SMALL_hdx_vs_E"].GetXaxis().GetBinLowEdge(ie)
        Emax = histos["SMALL_hdx_vs_E"].GetXaxis().GetBinUpEdge(ie)
        for ix in range(1,histos["SMALL_hdx_vs_E"].GetNbinsY()+1):
            label_dx = str(ix)
            dxmin = histos["SMALL_hdx_vs_E"].GetYaxis().GetBinLowEdge(ix)
            dxmax = histos["SMALL_hdx_vs_E"].GetYaxis().GetBinUpEdge(ix)
            label = "E"+label_E+"_dx"+label_dx
            histos.update({"hdE_"+label: TH1D("hdE_"+label,label+";#DeltaE [MeV];Steps", len(bins.dEbins_small)-1,array.array("d",bins.dEbins_small))})
            # histos.update({"hdE_"+label: TH1D("hdE_"+label,label+";#DeltaE [MeV];Steps", 50000,bins.dEmin,bins.dEmax)})
            histos.update({"hE_"+label:  TH1D("hE_"+label,label+";E [MeV];Steps", bins.n_small_E,Emin,Emax)})
            histos.update({"hdx_"+label: TH1D("hdx_"+label,label+";dx [#mum];Steps", bins.n_small_dx,dxmin,dxmax)})
    
    histos.update({"SMALL_hdxinv_vs_E" : TH2D("SMALL_h_dxinv_vs_E",";E [MeV];1/dx [1/#mum];Steps", bins.n_small_E,bins.Emin,bins.Emax, len(bins.dxinvbins_small)-1,array.array("d",bins.dxinvbins_small))})
    for ie in range(1,histos["SMALL_hdxinv_vs_E"].GetNbinsX()+1):
        label_E = str(ie)
        Emin = histos["SMALL_hdxinv_vs_E"].GetXaxis().GetBinLowEdge(ie)
        Emax = histos["SMALL_hdxinv_vs_E"].GetXaxis().GetBinUpEdge(ie)
        histos.update({"hMPVratio_E"+label_E: TH1D("hMPVratio_E"+label_E,label_E+";dx [#mum];Theory_{MPV}/Hist_{MPV}",len(bins.dxbins_small)-1,bins.dxbins_small)})
        histos.update({"hMeanratio_E"+label_E: TH1D("hMeanratio_E"+label_E,label_E+";dx [#mum];Theory_{Mean}/Hist_{Mean}",len(bins.dxbins_small)-1,bins.dxbins_small)})
        histos.update({"hN1_E"+label_E: TH1D("hN1_E"+label_E,label_E+";dx [#mum];Number of collisions of type 1",len(bins.dxbins_small)-1,bins.dxbins_small)})
        histos.update({"hN3_E"+label_E: TH1D("hN3_E"+label_E,label_E+";dx [#mum];Number of collisions of type 3",len(bins.dxbins_small)-1,bins.dxbins_small)})
        histos.update({"hN0_E"+label_E: TH1D("hN0_E"+label_E,label_E+";dx [#mum];Number of collisions of type 0",len(bins.dxbins_small)-1,bins.dxbins_small)})
        for ixinv in range(1,histos["SMALL_hdxinv_vs_E"].GetNbinsY()+1):
            label_dxinv = str(ixinv)
            dxinvmin = histos["SMALL_hdxinv_vs_E"].GetYaxis().GetBinLowEdge(ixinv)
            dxinvmax = histos["SMALL_hdxinv_vs_E"].GetYaxis().GetBinUpEdge(ixinv)
            label = "E"+label_E+"_dxinv"+label_dxinv
            histos.update({"hdE_"+label: TH1D("hdE_"+label,label+";#DeltaE [MeV];Steps", len(bins.dEbins_small)-1,array.array("d",bins.dEbins_small))})
            # histos.update({"hdE_"+label: TH1D("hdE_"+label,label+";#DeltaE [MeV];Steps", 50000,bins.dEmin,bins.dEmax)})
            histos.update({"hE_"+label:  TH1D("hE_"+label,label+";E [MeV];Steps", bins.n_small_E,Emin,Emax)})
            histos.update({"hdxinv_"+label: TH1D("hdxinv_"+label,label+";1/dx [1/#mum];Steps", bins.n_small_dxinv,dxinvmin,dxinvmax)})
            histos.update({"hdx_"+label: TH1D("hdx_"+label,label+";dx [#mum];Steps", bins.n_small_dx,1./dxinvmax,1./dxinvmin)})
