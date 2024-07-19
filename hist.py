import math
import array
import numpy as np
import ROOT
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
        gridx.append( ROOT.TLine(X,Ymin,X,Ymax) )
    for x in range(1,h.GetNbinsY()+1):
        Y    = h.GetYaxis().GetBinLowEdge(x)
        Xmin = h.GetXaxis().GetXmin()
        Xmax = h.GetXaxis().GetXmax()
        gridy.append( ROOT.TLine(Xmin,Y,Xmax,Y) )
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
    hAv   = ROOT.TH1D(name,title,nbins,xmin,xmax) if(not isLogx) else ROOT.TH1D(name,title,len(xbins)-1,xbins)
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


def find_h_max(h,firstbin=1):
    hmax = -1e20
    for b in range(firstbin,h.GetNbinsX()+1):
        y = h.GetBinContent(b)
        if(y>hmax): hmax = y
    return hmax

def reset_hrange_left(h,xmin):
    hclone = h.Clone(h.GetName()+"_clone")
    hclone.GetXaxis().SetLimits(xmin,hclone.GetXaxis().GetXmax())
    return hclone


def hlin_truncate_negative(h,x0=1e-6):
    if(h.GetBinWidth(2)!=h.GetBinWidth(1)):
        print("hlin_truncate_negative can truncate only linearly binned histograms. quitting.")
        quit()
    n2truncate = 0
    for b in range(1,h.GetNbinsX()+1):
        if(h.GetXaxis().GetBinLowEdge(b)<=x0): n2truncate += 1
        else:                                  break
    nbefore = h.GetNbinsX()
    nafter  = nbefore-n2truncate
    bin1    = n2truncate+1
    xmin    = h.GetXaxis().GetBinLowEdge(bin1)
    xmax    = h.GetXaxis().GetXmax()
    xtitle  = h.GetXaxis().GetTitle()
    ytitle  = h.GetYaxis().GetTitle()
    title   = h.GetTitle()
    name    = h.GetName()+"_truncated"
    hnew    = ROOT.TH1D(name,title+";"+xtitle+";"+ytitle,nafter,xmin,xmax)
    
    print(f"n2truncate={n2truncate}, bin1={bin1}, xmin={xmin}, xmax={xmax}")
    
    for b in range(1,hnew.GetNbinsX()+1):
        if((h.GetBinCenter(b+n2truncate)-hnew.GetBinCenter(b))>1e-6):
            print("hlin_truncate_negative bin centers do not match. quitting.")
            quit()
        y = h.GetBinContent(b+n2truncate)
        hnew.SetBinContent(b,y)
    hnew.SetLineColor(h.GetLineColor())
    hnew.SetLineWidth(h.GetLineWidth())
    hnew.SetLineStyle(h.GetLineStyle())
    return hnew
    

def book(histos,emin=-1): ### must pass by reference!
    if(emin>0): bins.Emin = emin
    histos.update({"hE":           ROOT.TH1D("h_E",";E [MeV];Steps", 1000,bins.Emin,bins.Emax)})
    histos.update({"hdE":          ROOT.TH1D("h_dE",";#DeltaE [MeV];Steps", len(bins.dEbins)-1,array.array("d",bins.dEbins))})
    histos.update({"hdE_cnt":      ROOT.TH1D("h_dE_cnt",";#DeltaE [MeV];Steps", len(bins.dEbins)-1,array.array("d",bins.dEbins))})
    histos.update({"hdE_sec":      ROOT.TH1D("h_dE_sec",";#DeltaE [MeV];Steps", len(bins.dEbins)-1,array.array("d",bins.dEbins))})
    histos.update({"hdx":          ROOT.TH1D("h_dx",";dx [#mum];Steps", len(bins.dxbins)-1,array.array("d",bins.dxbins))})
    histos.update({"hdxinv":       ROOT.TH1D("h_dxinv",";1/dx [1/#mum];Steps", len(bins.dxinvbins)-1,array.array("d",bins.dxinvbins))})
    histos.update({"hdR":          ROOT.TH1D("h_dR",";dR [#mum];Steps", len(bins.dRbins)-1,array.array("d",bins.dRbins))})
    histos.update({"hdRinv":       ROOT.TH1D("h_dRinv",";1/dR [1/#mum];Steps", len(bins.dRinvbins)-1,array.array("d",bins.dRinvbins))})
    histos.update({"hdL":          ROOT.TH1D("h_dL",";dL [#mum];Steps", len(bins.dLbins)-1,array.array("d",bins.dLbins))})
    
    histos.update({"hdR_vs_dx":    ROOT.TH2D("hdR_vs_dx",";dx [#mum];dR [#mum];Steps", len(bins.dxbins)-1,array.array("d",bins.dxbins),len(bins.dRbins)-1,array.array("d",bins.dRbins))})
    
    histos.update({"hdEdx" :       ROOT.TH1D("h_dEdx",";dE/dx [MeV/#mum];Steps", len(bins.dEdxbins)-1,array.array("d",bins.dEdxbins))})
    histos.update({"hdEdx_cnt" :   ROOT.TH1D("h_dEdx_cnt",";dE/dx [MeV/#mum];Steps", len(bins.dEdxbins)-1,array.array("d",bins.dEdxbins))})
    histos.update({"hdEdx_sec" :   ROOT.TH1D("h_dEdx_sec",";dE/dx [MeV/#mum];Steps", len(bins.dEdxbins)-1,array.array("d",bins.dEdxbins))})
    histos.update({"hdEdx_vs_E":   ROOT.TH2D("h_dEdx_vs_E",";E [MeV];dE/dx [MeV/#mum];Steps",500,bins.Emin,bins.Emax, len(bins.dEdxbins)-1,array.array("d",bins.dEdxbins))})
    histos.update({"hdEdx_vs_E_small":     ROOT.TH2D("h_dEdx_vs_E_small",";E [MeV];dE/dx [MeV/#mum];Steps",len(bins.Ebins_small)-1,bins.Ebins_small, len(bins.dEdxbins)-1,array.array("d",bins.dEdxbins))})
    histos.update({"hdEdx_vs_E_small_cnt": ROOT.TH2D("h_dEdx_vs_E_small_cnt",";E [MeV];dE/dx [MeV/#mum];Steps",len(bins.Ebins_small)-1,bins.Ebins_small, len(bins.dEdxbins)-1,array.array("d",bins.dEdxbins))})
    histos.update({"hdEdx_vs_E_small_sec": ROOT.TH2D("h_dEdx_vs_E_small_sec",";E [MeV];dE/dx [MeV/#mum];Steps",len(bins.Ebins_small)-1,bins.Ebins_small, len(bins.dEdxbins)-1,array.array("d",bins.dEdxbins))})
    histos.update({"hdE_vs_dx":    ROOT.TH2D("h_dE_vs_dx",";dx [#mum];#DeltaE [MeV];Steps", len(bins.dxbins)-1,array.array("d",bins.dxbins), len(bins.dEbins)-1,array.array("d",bins.dEbins))})
    histos.update({"hdE_vs_dxinv": ROOT.TH2D("h_dE_vs_dxinv",";1/dx [1/#mum];#DeltaE [MeV];Steps", len(bins.dxinvbins)-1,array.array("d",bins.dxinvbins), len(bins.dEbins)-1,array.array("d",bins.dEbins))})
    histos.update({"hdx_vs_E":     ROOT.TH2D("h_dx_vs_E",";E [MeV];dx [#mum];Steps", 500,bins.Emin,bins.Emax, len(bins.dxbins)-1,array.array("d",bins.dxbins))})
    histos.update({"hdxinv_vs_E":  ROOT.TH2D("h_dxinv_vs_E",";E [MeV];1/dx [1/#mum];Steps", 500,bins.Emin,bins.Emax, len(bins.dxinvbins)-1,array.array("d",bins.dxinvbins))})
    histos.update({"hdL_vs_E":     ROOT.TH2D("h_dL_vs_E",";E [MeV];#DeltaL [#mum];Steps", 500,bins.Emin,bins.Emax, len(bins.dLbins)-1,array.array("d",bins.dLbins))})
    
    histos.update({"SMALL_hdx_vs_E_N1" : ROOT.TH2D("SMALL_hdx_vs_E_N1",";E [MeV];dx [#mum];<n_{1}>", bins.n_small_E,bins.Emin,bins.Emax, len(bins.dxbins_small)-1,array.array("d",bins.dxbins_small))})
    histos.update({"SMALL_hdx_vs_E_N3" : ROOT.TH2D("SMALL_hdx_vs_E_N3",";E [MeV];dx [#mum];<n_{3}>", bins.n_small_E,bins.Emin,bins.Emax, len(bins.dxbins_small)-1,array.array("d",bins.dxbins_small))})
    histos.update({"SMALL_hdx_vs_E_N0" : ROOT.TH2D("SMALL_hdx_vs_E_N0",";E [MeV];dx [#mum];<n_{0}>", bins.n_small_E,bins.Emin,bins.Emax, len(bins.dxbins_small)-1,array.array("d",bins.dxbins_small))})
    
    histos.update({"SMALL_hdx_vs_E_isGauss_N1" : ROOT.TH2D("SMALL_hdx_vs_E_isGauss_N1",";E [MeV];dx [#mum];Gaussian for <n_{1}>", bins.n_small_E,bins.Emin,bins.Emax, len(bins.dxbins_small)-1,array.array("d",bins.dxbins_small))})
    histos.update({"SMALL_hdx_vs_E_isGauss_N3" : ROOT.TH2D("SMALL_hdx_vs_E_isGauss_N3",";E [MeV];dx [#mum];Gaussian for <n_{3}>", bins.n_small_E,bins.Emin,bins.Emax, len(bins.dxbins_small)-1,array.array("d",bins.dxbins_small))})
    histos.update({"SMALL_hdx_vs_E_isGauss_N0" : ROOT.TH2D("SMALL_hdx_vs_E_isGauss_N0",";E [MeV];dx [#mum];Gaussian for <n_{0}>", bins.n_small_E,bins.Emin,bins.Emax, len(bins.dxbins_small)-1,array.array("d",bins.dxbins_small))})
    
    histos.update({"SMALL_hdL_vs_E"    : ROOT.TH2D("SMALL_h_dL_vs_E",";E [MeV];#DeltaL [#mum];Steps",        bins.n_small_E,bins.Emin,bins.Emax, len(bins.dLbins_small)-1,array.array("d",bins.dLbins_small))})    
    histos.update({"SMALL_hdx_vs_E"    : ROOT.TH2D("SMALL_h_dx_vs_E",";E [MeV];#Deltax [#mum];Steps",        bins.n_small_E,bins.Emin,bins.Emax, len(bins.dxbins_small)-1,array.array("d",bins.dxbins_small))})    
    histos.update({"SMALL_hdxinv_vs_E" : ROOT.TH2D("SMALL_h_dxinv_vs_E",";E [MeV];1/#Deltax [1/#mum];Steps", bins.n_small_E,bins.Emin,bins.Emax, len(bins.dxinvbins_small)-1,array.array("d",bins.dxinvbins_small))})

