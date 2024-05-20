import math
import array
import numpy as np
import ROOT
import units as U
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq, rfft, irfft
from scipy.special import sici, exp1
from scipy.signal import convolve, fftconvolve

# ROOT.gROOT.SetBatch(1)
# ROOT.gStyle.SetOptFit(0)
# ROOT.gStyle.SetOptStat(0)
ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning


### this has to stay outside of any class
### so it can be called to construct a TF1
def borysov_excitation(x,par):
    ## par[0] = n1 (mean number of collisions of type 1)
    ## par[1] = e1 (type 1 excitation energy)
    nn = int(x[0]/par[1])
    if(x[0]<0): nn -= 1
    xx = 0
    nnmax = nn+2
    for ii in range(nn,nnmax):
        if(ii<=0): continue
        # if(ii<0): continue ### WRONG
        xx += math.pow(par[0],ii) / ROOT.TMath.Factorial(ii)
    xx *= 0.5*ROOT.TMath.Exp(-par[0])/par[1] #if(x[0]>0) else 0
    return xx

### this has to stay outside of any class
### so it can be called to construct a TF1
def truncated_gaus(x,par):
    ## par[0] = mean
    ## par[1] = sigma
    return ROOT.TMath.Gaus(x[0],par[0],par[1]) if(x[0]>0 and x[0]<2*par[0]) else 0


class Model:
    def __init__(self,build,dx,E,minLoss,meanLoss,w3,p3,w,e1,n1,ex1_mean,ex1_sigma,ion_mean,ion_sigma):
        self.BEBL      = False
        self.IONB      = False
        self.EX1B      = False
        self.IONG      = False
        self.dx        = dx ## cm
        self.dx_um     = dx*U.cm2um # um
        self.E         = E  ## eV
        self.build     = build
        self.minLoss   = minLoss
        self.meanLoss  = meanLoss
        self.w3        = w3
        self.p3        = p3
        self.w         = w
        self.e1        = e1
        self.n1        = n1
        self.ex1_mean  = ex1_mean
        self.ex1_sigma = ex1_sigma
        self.ion_mean  = ion_mean
        self.ion_sigma = ion_sigma
        ### set parameters
        self.par_borysov_ion = [self.w3, self.w, self.p3]
        self.par_borysov_exc = [self.n1, self.e1]
        self.par_gauss_ion   = [self.ion_mean, self.ion_sigma]
        self.par_gauss_exc   = [self.ex1_mean, self.ex1_sigma]
        
        self.NptsTF1    = 1000000
        self.N_t_bins   = 1000000
        self.dEmin      = -1
        self.dEmax      = -1
        self.Nbins      = -1
        self.doLogx     = False
        self.plotPsi    = True
        self.convManual = True
        self.convMode   = "full"
        self.psiRe      = None
        self.psiIm      = None
        
        ### intialize everything else
        self.set_flags()
        self.validate_pars()
        self.dE_binning()
        ### for the frequencies spacing
        self.tmin = -1
        self.tmax = -1
        self.fSampling = -1
        self.TSampling = -1
        if(self.IONB): self.set_fft_sampling_pars(self.par_borysov_ion)
        
    
    ### get the flags from the build string
    def set_flags(self):
        self.BEBL = (self.meanLoss<self.minLoss)
        self.IONB = ("ION.B" in self.build)
        self.EX1B = ("EX1.B" in self.build)
        self.IONG = ("ION.G" in self.build)
        self.EX1G = ("EX1.G" in self.build)
        print(f"BEBL={self.BEBL}, IONB={self.IONB}, EX1B={self.EX1B}, IONG={self.IONG}, EX1G={self.EX1G}")

    ### make sure the parameters are passed correctly
    def validate_pars(self):
        ## very small losses
        if(not self.BEBL): self.meanLoss = -1
        if(not self.BEBL): self.minLoss  = -1
        ## ionization non-gauss
        if(not self.IONB): self.w3 = -1
        if(not self.IONB): self.p3 = -1
        if(not self.IONB): self.w  = -1
        ## excitation non-gauss
        if(not self.EX1B): self.e1 = -1
        if(not self.EX1B): self.n1 = -1
        ## excitation gauss
        if(not self.EX1G): self.ex1_mean  = -1
        if(not self.EX1G): self.ex1_sigma = -1
        ## ionization gauss
        if(not self.IONG): self.ion_mean  = -1
        if(not self.IONG): self.ion_sigma = -1
    
    def dE_binning(self):
        if(self.BEBL):
            self.dEmin     = 0
            self.dEmax     = 11
            self.Nbins     = 110
        if(self.IONB and self.EX1B and not self.IONG and not self.EX1G): ## Borysov only, no Gauss
            self.dEmin     = 0.1
            self.dEmax     = 3000.1
            self.Nbins     = 6000
        if(self.IONB and not self.EX1B and self.IONG and self.EX1G): ## no Borysov Exc
            self.dEmin     = -3000
            self.dEmax     = 53000
            self.Nbins     = 12500
        if(self.IONB and (self.IONG and not self.EX1G) or (self.EX1G and not self.IONG)): ## only one Gauss
            self.dEmin     = -2000
            self.dEmax     = 12000
            self.Nbins     = 7000
        self.doLogx = True if(self.dEmin>0) else False
        print(f"dEmin={self.dEmin}, dEmax={self.dEmax}, Nbins={self.Nbins}")
    
    def set_fft_sampling_pars(self,par):
        # for the frequencies spacing
        if(self.dx_um>1.0): ## in microns
            self.tmin = -2
            self.tmax = +2
        else:
            self.tmin = -50
            self.tmax = +50
        self.fSampling = (2*np.pi)*(self.N_t_bins/(self.tmax-self.tmin))
        self.TSampling = 1./self.fSampling
        print(f"tmin={self.tmin}, tmax={self.tmax}, TSampling={self.TSampling}")

    def scipy_psi_of_t_as_h(self,name,par):
        ## par[0] = w3
        ## par[1] = w
        ## par[2] = p3 (lambda)
        ## par[3] = 0/1 Re/Im
        a = par[0]
        b = par[0]/(1-par[1])
        t = np.linspace(self.tmin,self.tmax,self.N_t_bins)
        aSi, aCi = sici(a*t)
        bSi, bCi = sici(b*t)
        A = np.cos(a*t)/a - np.cos(b*t)/b + t*aSi - t*bSi
        B = -t*aCi + t*bCi + np.sin(a*t)/a - np.sin(b*t)/b
        C = par[2]*par[0]/par[1]
        psi = (1./(2*np.pi))*np.exp(C*A-par[2])*( np.cos(C*B) + 1j*np.sin(C*B) )
        psi_re = np.real( psi )
        psi_im = np.imag( psi )
        h_re = ROOT.TH1D("h_re_"+name,"Borysov Re[#psi(t)];t [1/eV];Re[#psi(t)]",self.N_t_bins,self.tmin,self.tmax)
        h_im = ROOT.TH1D("h_im_"+name,"Borysov Im[#psi(t)];t [1/eV];Im[#psi(t)]",self.N_t_bins,self.tmin,self.tmax)
        for bb in range(1,h_re.GetNbinsX()+1):
            x = h_re.GetBinCenter(bb)
            y_re = psi_re[bb-1]
            y_im = psi_im[bb-1]
            h_re.SetBinContent(bb,y_re)
            h_im.SetBinContent(bb,y_im)
        h_re.Scale(1./h_re.Integral())
        h_im.Scale(1./h_im.Integral())
        h_re.SetLineColor( ROOT.kRed )
        h_im.SetLineColor( ROOT.kBlue )
        h_re.SetLineWidth( 1 )
        h_im.SetLineWidth( 1 )
        return h_re,h_im

    def borysov_ionization(self,par):
        ### get psi(t)
        self.psiRe, self.psiIm = self.scipy_psi_of_t_as_h("psi_of_t",par)
        ### The FFT
        y_lst = []
        for bb in range(1,self.N_t_bins+1):
            re = self.psiRe.GetBinContent(bb)
            im = self.psiIm.GetBinContent(bb)
            y_lst.append( re + im*1.j )
        y = np.array(y_lst)
        yf = fft(y)
        xf = fftfreq(self.N_t_bins, self.TSampling)[:self.N_t_bins//2] ## remove the last M elements, where M=floor(N/2)
        ya = np.abs(yf[0:self.N_t_bins//2])*(2/self.N_t_bins)
        ### Get the integral as graph
        gFFT = ROOT.TGraph(len(xf),xf,ya)
        gFFT.SetLineColor( ROOT.kRed )
        return gFFT

    def get_pdf(self,name,pdfname,par,dE_lowcut=-1):
        title = name.replace("_model","")
        f = None
        g = None
        isBorysovIon = (pdfname=="borysov_ionization")
        if(pdfname=="borysov_ionization"): g = self.borysov_ionization(par) ### This is a graph!
        if(pdfname=="borysov_excitation"): f = ROOT.TF1("f_"+name,borysov_excitation,self.dEmin,self.dEmax,len(par))
        if(pdfname=="truncated_gaus"):     f = ROOT.TF1("f_"+name,truncated_gaus,self.dEmin,self.dEmax,len(par))
        if(isBorysovIon):
            g.SetLineColor(ROOT.kRed)
        else:
            for i in range(len(par)): f.SetParameter(i,par[i])
            f.SetNpx(self.NptsTF1)
            f.SetLineColor(ROOT.kRed)
        h = ROOT.TH1D("h_"+name,"h_"+title,self.Nbins,self.dEmin,self.dEmax)
        for bb in range(1,h.GetNbinsX()+1):
            x = h.GetBinCenter(bb)
            y = g.Eval(x) if(isBorysovIon) else f.Eval(x)
            if(x<dE_lowcut and dE_lowcut>0): y = 0 #TODO: is this cutoff physical?
            if(x<=0 and isBorysovIon):       y = 0 #TODO: is this OK?
            h.SetBinContent(bb,y)
        h.Scale(1./h.Integral())
        h.SetLineColor( ROOT.kRed )
        h.SetLineWidth( 1 )
        return h

    def manual_convolution(self,A,K):
        aManualConv = []
        for k in range(len(K)):
            S = 0
            for i in range(len(A)):
                if(i>k): continue
                S += A[i]*K[k-i]
            aManualConv.append(S)
        aManualConv = np.array(aManualConv)
        return aManualConv
    
    def get_component_pdfs(self):
        ### get pdfs
        pdfs = {}
        pdfs.update({"hBorysov_Ion":  None})
        pdfs.update({"hBorysov_Exc":  None})
        pdfs.update({"hTrncGaus_Ion": None})
        pdfs.update({"hTrncGaus_Exc": None})
        if(self.IONB and self.EX1B and self.IONG):
            pdfs["hBorysov_Ion"]  = self.get_pdf("borysov_ion_model", "borysov_ionization", self.par_borysov_ion)
            pdfs["hBorysov_Exc"]  = self.get_pdf("borysov_exc_model", "borysov_excitation", self.par_borysov_exc)
            pdfs["hTrncGaus_Ion"] = self.get_pdf("gauss_ion_model",   "truncated_gaus",     self.par_gauss_ion)
            print(f'PDF Integrals: hBorysov_Ion={pdfs["hBorysov_Ion"].Integral()}, hBorysov_Exc={pdfs["hBorysov_Exc"].Integral()}, hTrncGaus_Ion={pdfs["hTrncGaus_Ion"].Integral()}')
        if(self.IONB and self.IONG and self.EX1G):
            pdfs["hBorysov_Ion"]  = self.get_pdf("borysov_ion_model", "borysov_ionization", self.par_borysov_ion)
            pdfs["hTrncGaus_Ion"] = self.get_pdf("gauss_ion_model",   "truncated_gaus",     self.par_gauss_ion)
            pdfs["hTrncGaus_Exc"] = self.get_pdf("gauss_exc_model",   "truncated_gaus",     self.par_gauss_exc)
            print(f'PDF Integrals: hBorysov_Ion={pdfs["hBorysov_Ion"].Integral()}, hTrncGaus_Ion={pdfs["hTrncGaus_Ion"].Integral()}, hTrncGaus_Exc={pdfs["hTrncGaus_Exc"].Integral()}')
        if(self.IONB and self.EX1B and not self.IONG and not self.EX1G):
            pdfs["hBorysov_Ion"] = self.get_pdf("borysov_ion_model", "borysov_ionization", self.par_borysov_ion)
            pdfs["hBorysov_Exc"] = self.get_pdf("borysov_exc_model", "borysov_excitation", self.par_borysov_exc)
            print(f'PDF Integrals: hBorysov_Ion={pdfs["hBorysov_Ion"].Integral()}, hBorysov_Exc={pdfs["hBorysov_Exc"].Integral()}')
        return pdfs
    
    def get_model_pdfs(self):
        pdfs = self.get_component_pdfs()
        aBorysov_Ion  = []
        aBorysov_Exc  = []
        aTrncGaus_Ion = []
        aTrncGaus_Exc = []
        if(not self.BEBL):
            for b in range(1,pdfs["hBorysov_Ion"].GetNbinsX()+1):
                if(pdfs["hBorysov_Ion"]  is not None): aBorysov_Ion.append(pdfs["hBorysov_Ion"].GetBinContent(b))
                if(pdfs["hBorysov_Exc"]  is not None): aBorysov_Exc.append(pdfs["hBorysov_Exc"].GetBinContent(b))
                if(pdfs["hTrncGaus_Ion"] is not None): aTrncGaus_Ion.append(pdfs["hTrncGaus_Ion"].GetBinContent(b))
                if(pdfs["hTrncGaus_Exc"] is not None): aTrncGaus_Exc.append(pdfs["hTrncGaus_Exc"].GetBinContent(b))
        aBorysov_Ion  = np.array(aBorysov_Ion)
        aBorysov_Exc  = np.array(aBorysov_Exc)
        aTrncGaus_Ion = np.array(aTrncGaus_Ion)
        aTrncGaus_Exc = np.array(aTrncGaus_Exc)

        aScipyConv  = None
        aManualConv = None
        if(self.IONB and self.EX1B and self.IONG):
            if(self.convManual):
                aManualConv1 = self.manual_convolution(aBorysov_Ion,aBorysov_Exc)
                aManualConv2 = self.manual_convolution(aTrncGaus_Ion,aManualConv1)
                aManualConv = aManualConv2
            else:
                aScipyConv1 = convolve(aBorysov_Ion,aBorysov_Exc, mode=self.convMode, method='auto')
                aScipyConv2 = convolve(aTrncGaus_Ion,aScipyConv1, mode=self.convMode, method='auto')
                aScipyConv = aScipyConv2
            print(f"sizes of input arrays for IONB={len(aBorysov_Ion)}, EX1B={len(aBorysov_Exc)}, IONG={len(aTrncGaus_Ion)}")
            print(f"sizes of convolutions for (IONB and EX1B and IONG): IONBxEX1B={len(aManualConv1) if(self.convManual) else len(aScipyConv1)}, IONBxEX1BxIONG={len(aManualConv2) if(self.convManual) else len(aScipyConv2)}")

        if(self.IONB and self.IONG and self.EX1G):
            if(self.convManual):
                aManualConv1 = self.manual_convolution(aBorysov_Ion,aTrncGaus_Ion)
                aManualConv2 = self.manual_convolution(aTrncGaus_Exc,aManualConv1)
                aManualConv = aManualConv2
            else:
                aScipyConv1 = convolve(aBorysov_Ion,aTrncGaus_Ion, mode=convMode, method='auto')
                aScipyConv2 = convolve(aTrncGaus_Exc,aScipyConv1,  mode=convMode, method='auto')
                aScipyConv = aScipyConv2
            print(f"sizes of input arrays for IONB={len(aBorysov_Ion)}, IONG={len(aTrncGaus_Ion)}, EX1G={len(aTrncGaus_Exc)}")
            print(f"sizes of convolutions for (IONB and IONG and EX1G): IONBxIONG={len(aManualConv1) if(self.convManual) else len(aScipyConv1)}, IONBxIONGxEX1G={len(aManualConv2) if(self.convManual) else len(aScipyConv2)}")

        if(self.IONB and self.EX1B and not self.IONG and not self.EX1G):
            if(self.convManual):
                aManualConv1 = self.manual_convolution(aBorysov_Ion,aBorysov_Exc)
                aManualConv = aManualConv1
            else:
                aScipyConv1 = convolve(aBorysov_Ion,aBorysov_Exc, mode=convMode, method='auto')
                aScipyConv = aScipyConv1
            print(f"sizes of input arrays for IONB={len(aBorysov_Ion)}, EX1B={len(aBorysov_Exc)}")
            print(f"sizes of convolutions for (IONB and EX1B and not IONG and not EX1G): IONBxEX1B={len(aManualConv1) if(self.convManual) else len(aScipyConv1)}")

        if(not self.BEBL):
            aConv = aManualConv if(self.convManual) else aScipyConv
            hModel = pdfs["hBorysov_Ion"].Clone("hModel")
            hModel.Reset()
            hModel.SetLineColor(ROOT.kRed)
            xConv = np.linspace(start=self.dEmin,stop=self.dEmax,num=len(aConv))
            gConv = ROOT.TGraph(len(aConv),xConv, aConv)
            for b in range(1,hModel.GetNbinsX()+1):
                # hModel.SetBinContent(b, aManualConv[b-1] if(self.convManual) else aScipyConv[b-1])
                xb = hModel.GetBinCenter(b)
                hModel.SetBinContent(b, gConv.Eval(xb+2*abs(self.dEmin)) )
            hModel.Scale(1./hModel.Integral())
            print(f"hModel={hModel.GetNbinsX()}, aConv={len(aConv)}")
        else:
            hModel = hTotal.Clone("hModel")
            hModel.Reset()
            hModel.SetLineColor(ROOT.kRed)
            bx = hModel.FindBin(meanLoss)
            hModel.SetBinContent(bx,1)
        pdfs.update({"hModel":hModel})
        return pdfs