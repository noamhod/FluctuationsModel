import math
import array
import numpy as np
import ROOT
import units as U
from scipy.fft import fft, fftfreq, rfft, irfft
from scipy.special import sici, exp1
from scipy.signal import convolve, fftconvolve
import time

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


### borysov_secondaries main function (see above)
### so it can be called to construct a TF1
def inv_sum_distribution(x,par):
    res = 0.0
    A = par[1] #TODO: there's a weird swap between A and B
    B = par[0] #TODO: there's a weird swap between A and B
    X = x[0]
    if(abs(X)>1.0e-10):
        xx = A*B/X
        if(xx>0. and xx<=B):   res = xx/(A*B)
        if(xx>B  and xx<=A):   res = 1.0/A
        if(xx>A  and xx<=A+B): res = (A+B-xx)/(A*B)
    return res*A*B/(X**2)


class Model:
    def __init__(self,dx,E,pars,dotime=False):
        self.dotime    = dotime
        self.SECB      = False
        self.TGAU      = False
        self.TGAM      = False
        self.BEBL      = False
        self.IONB      = False
        self.EX1B      = False
        self.IONG      = False
        self.EX1G      = False
        self.dx        = dx ## cm
        self.dx_um     = dx*U.cm2um # um
        self.E         = E  ## eV
        self.build     = pars["build"]
        self.scale     = pars["scale"]
        self.param     = pars["param"]
        self.spin      = self.param["spin"]      if("spin"      in self.param) else -1
        self.minLoss   = self.param["minLoss"]   if("minLoss"   in self.param) else -1
        self.meanLoss  = self.param["meanLoss"]  if("meanLoss"  in self.param) else -1
        self.Tmax      = self.param["Tmax"]      if("Tmax"      in self.param) else -1
        self.Etot      = self.param["Etot"]      if("Etot"      in self.param) else -1
        self.b2        = self.param["b2"]        if("b2"        in self.param) else -1
        self.EkinMin   = self.param["EkinMin"]   if("EkinMin"   in self.param) else -1
        self.EkinMax   = self.param["EkinMax"]   if("EkinMax"   in self.param) else -1
        self.fmax      = self.param["fmax"]      if("fmax"      in self.param) else -1
        self.w3        = self.param["w3"]        if("w3"        in self.param) else -1
        self.p3        = self.param["p3"]        if("p3"        in self.param) else -1
        self.w         = self.param["w"]         if("w"         in self.param) else -1
        self.e1        = self.param["e1"]        if("e1"        in self.param) else -1
        self.n1        = self.param["n1"]        if("n1"        in self.param) else -1
        self.ex1_mean  = self.param["ex1_mean"]  if("ex1_mean"  in self.param) else -1
        self.ex1_sigma = self.param["ex1_sigma"] if("ex1_sigma" in self.param) else -1
        self.ion_mean  = self.param["ion_mean"]  if("ion_mean"  in self.param) else -1
        self.ion_sigma = self.param["ion_sigma"] if("ion_sigma" in self.param) else -1
        ### set parameters
        self.par_bethebloch_min  = [self.meanLoss]
        self.par_borysov_sec     = [self.EkinMin, self.EkinMax]
        self.par_borysov_ion     = [self.w3, self.w, self.p3]
        self.par_borysov_exc     = [self.n1, self.e1]
        self.par_gauss_ion       = [self.ion_mean, self.ion_sigma]
        self.par_gauss_exc       = [self.ex1_mean, self.ex1_sigma]
        
        self.NptsTF1    = 1000000
        self.N_t_bins   = 1000000
        self.dEmin      = -1
        self.dEmax      = -1
        self.Nbins      = -1
        self.dEminSec   = -1
        self.dEmaxSec   = -1
        self.NbinsSec   = -1
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
        self.SECB = ("SEC.B" in self.build)
        self.TGAU = ("THK.GAUSS" in self.build)
        self.TGAM = ("THK.GAMMA" in self.build)
        self.BEBL = (self.meanLoss<self.minLoss and "BEBL" in self.build)
        self.IONB = ("ION.B" in self.build)
        self.EX1B = ("EX1.B" in self.build)
        self.IONG = ("ION.G" in self.build)
        self.EX1G = ("EX1.G" in self.build)
        print(f"SECB={self.SECB}, BEBL={self.BEBL}, IONB={self.IONB}, EX1B={self.EX1B}, IONG={self.IONG}, EX1G={self.EX1G}")

    ### make sure the parameters are passed correctly
    def validate_pars(self):
        ## very small losses
        if(not self.SECB): self.EkinMin = -1
        if(not self.SECB): self.EkinMax = -1
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
        if(self.SECB):
            self.dEminSec  = 10
            self.dEmaxSec  = 1000000.1
            self.NbinsSec  = 20000
        if(self.BEBL):
            self.dEmin     = 0
            self.dEmax     = 11
            self.Nbins     = 1100
        if(self.IONB and self.EX1B and not self.IONG and not self.EX1G): ## Borysov only, no Gauss
            self.dEmin     = 0.1
            self.dEmax     = 3000.1
            self.Nbins     = 6000
        if(self.IONB and not self.EX1B and self.IONG and self.EX1G): ## no Borysov Exc
            self.dEmin     = 0.1 #-3000
            self.dEmax     = 60000.1 #53000
            self.Nbins     = 15000 # 12500
        if(self.IONB and (self.IONG and not self.EX1G) or (self.EX1G and not self.IONG)): ## only one Gauss
            self.dEmin     = 0.1 #-2000
            self.dEmax     = 15000.1 # 12000
            self.Nbins     = 7500 #7000
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
        start = time.time()
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
        end = time.time()
        if(self.dotime):
            elapsed = end-start
            print(f"TIME of scipy_psi_of_t_as_h: {elapsed} [s]")
        return h_re,h_im

    def borysov_ionization(self,par):
        start = time.time()
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
        gFFT.SetBit(ROOT.TGraph.kIsSortedX)
        end = time.time()
        if(self.dotime):
            elapsed = end-start
            print(f"TIME of borysov_ionization: {elapsed} [s]")
        return gFFT

    def get_pdf(self,name,pdfname,par,dE_lowcut=-1):
        start = time.time()
        title = name.replace("_model","")
        h = ROOT.TH1D("h_"+name,"h_"+title,self.Nbins,self.dEmin,self.dEmax)
        if("sec" in name): h = ROOT.TH1D("h_"+name,"h_"+title,self.NbinsSec,self.dEminSec,self.dEmaxSec)
        if(self.BEBL):
            bx = h.FindBin(par[0]) ## par[0] is meanLoss... 
            h.SetBinContent(bx,1)
        else:
            f = None
            g = None
            isBorysovIon = (pdfname=="borysov_ionization")
            if(pdfname=="borysov_ionization"):  g = self.borysov_ionization(par) ### This is a graph!
            if(pdfname=="borysov_secondaries"): f = ROOT.TF1("f_"+name,inv_sum_distribution,self.dEminSec,self.dEmaxSec,len(par))
            if(pdfname=="borysov_excitation"):  f = ROOT.TF1("f_"+name,borysov_excitation,self.dEmin,self.dEmax,len(par))
            if(pdfname=="truncated_gaus"):      f = ROOT.TF1("f_"+name,truncated_gaus,self.dEmin,self.dEmax,len(par))
            if(isBorysovIon): g.SetLineColor(ROOT.kRed)
            else:
                for i in range(len(par)):  f.SetParameter(i,par[i])
                f.SetNpx(self.NptsTF1)
                f.SetLineColor(ROOT.kRed)
            for bb in range(1,h.GetNbinsX()+1):
                x = h.GetBinCenter(bb)
                y = g.Eval(x) if(isBorysovIon) else f.Eval(x)
                if(x<dE_lowcut and dE_lowcut>0): y = 0 #TODO: is this cutoff physical?
                if(x<=0 and isBorysovIon):       y = 0 #TODO: is this OK?
                h.SetBinContent(bb,y)
            ### TODO: implement the thick cases
        h.Scale(1./h.Integral())
        h.SetLineColor( ROOT.kRed )
        h.SetLineWidth( 1 )
        end = time.time()
        if(self.dotime):
            elapsed = end-start
            print(f"TIME of get_pdf({name}): {elapsed} [s]")
        return h

    # def manual_convolution(self,A,K):
    #     start = time.time()
    #     aManualConv = []
    #     for k in range(len(K)):
    #         S = 0
    #         for i in range(len(A)):
    #             if(i>k): continue
    #             S += A[i]*K[k-i]
    #         aManualConv.append(S)
    #     aManualConv = np.array(aManualConv)
    #     end = time.time()
    #     if(self.dotime):
    #         elapsed = end-start
    #         print(f"TIME of manual_convolution: {elapsed} [s]")
    #     return aManualConv
    
    # def manual_convolution(self,A,K):
    #     start = time.time()
    #     aManualConv = []
    #     for k in range(len(K)):
    #         S = 0
    #         for i in range(len(A)):
    #             S += A[i]*K[k-i]
    #             if(i>k): break
    #         aManualConv.append(S)
    #     aManualConv = np.array(aManualConv)
    #     end = time.time()
    #     if(self.dotime):
    #         elapsed = end-start
    #         print(f"TIME of manual_convolution: {elapsed} [s]")
    #     return aManualConv
    
    def manual_convolution(self,A,K):
        start = time.time()
        n = len(K)
        m = len(A)
        result = np.zeros(n)
        for k in range(n):
            result[k] = np.dot(A[:k+1], K[k::-1])
        end = time.time()
        if(self.dotime):
            elapsed = end-start
            print(f"TIME of manual_convolution: {elapsed} [s]")
        return result
    
    def get_component_pdfs(self):
        start = time.time()
        ### get pdfs
        pdfs = {}
        pdfs.update({"hModel":        None})
        pdfs.update({"hBorysov_Ion":  None})
        pdfs.update({"hBorysov_Exc":  None})
        pdfs.update({"hTrncGaus_Ion": None})
        pdfs.update({"hTrncGaus_Exc": None})
        # pdfs.update({"hGauss_Thk":    None})
        # pdfs.update({"hGamma_Thk":    None})
        pdfs.update({"hBEBL_Thn":     None})
        pdfs["hModel"] = ROOT.TH1D("hModel","",self.Nbins,self.dEmin,self.dEmax)
        pdfs["hModel"].SetLineColor(ROOT.kRed)
        if(self.BEBL):
            pdfs["hBEBL_Thn"] = self.get_pdf("bethebloch_min_model", "bethebloch_min_model", self.par_bethebloch_min)
            print(f'PDF Integral: hBEBL_Thn={pdfs["hBEBL_Thn"].Integral()}')
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
        end = time.time()
        if(self.dotime):
            elapsed = end-start
            print(f"TIME of get_component_pdfs: {elapsed} [s]")
        return pdfs
        
    def get_secondaries_pdfs(self):
        start = time.time()
        ### get pdfs
        pdfs = {}
        pdfs.update({"hBorysov_Sec":  None})
        if(self.SECB):
            pdfs["hBorysov_Sec"] = self.get_pdf("borysov_sec_model", "borysov_secondaries", self.par_borysov_sec)
        end = time.time()
        if(self.dotime):
            elapsed = end-start
            print(f"TIME of get_secondaries_pdfs: {elapsed} [s]")
        return pdfs
    
    
    ### get the relevant pdfs
    def get_model_pdfs(self):
        start = time.time()
        ### get the pdfs of the continuous part
        pdfs = self.get_component_pdfs()
        ### if meanLoss is too small return the meanLoss as single bin PDF
        if(self.BEBL):
            for b in range(1,pdfs["hBEBL_Thn"].GetNbinsX()+1): pdfs["hModel"].SetBinContent(b, pdfs["hBEBL_Thn"].GetBinContent(b) )
            return pdfs
        ### Otherwise, constructe the continuous model 
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
                aScipyConv1 = fftconvolve(aBorysov_Ion,aBorysov_Exc, mode=self.convMode)#, method='auto')
                aScipyConv2 = fftconvolve(aTrncGaus_Ion,aScipyConv1, mode=self.convMode)#, method='auto')
                aScipyConv = aScipyConv2
            print(f"sizes of input arrays for IONB={len(aBorysov_Ion)}, EX1B={len(aBorysov_Exc)}, IONG={len(aTrncGaus_Ion)}")
            print(f"sizes of convolutions for (IONB and EX1B and IONG): IONBxEX1B={len(aManualConv1) if(self.convManual) else len(aScipyConv1)}, IONBxEX1BxIONG={len(aManualConv2) if(self.convManual) else len(aScipyConv2)}")
        if(self.IONB and self.IONG and self.EX1G):
            if(self.convManual):
                aManualConv1 = self.manual_convolution(aBorysov_Ion,aTrncGaus_Ion)
                aManualConv2 = self.manual_convolution(aTrncGaus_Exc,aManualConv1)
                aManualConv = aManualConv2
            else:
                aScipyConv1 = fftconvolve(aBorysov_Ion,aTrncGaus_Ion, mode=self.convMode)#, method='auto')
                aScipyConv2 = fftconvolve(aTrncGaus_Exc,aScipyConv1,  mode=self.convMode)#, method='auto')
                aScipyConv = aScipyConv2
            print(f"sizes of input arrays for IONB={len(aBorysov_Ion)}, IONG={len(aTrncGaus_Ion)}, EX1G={len(aTrncGaus_Exc)}")
            print(f"sizes of convolutions for (IONB and IONG and EX1G): IONBxIONG={len(aManualConv1) if(self.convManual) else len(aScipyConv1)}, IONBxIONGxEX1G={len(aManualConv2) if(self.convManual) else len(aScipyConv2)}")
        if(self.IONB and self.EX1B and not self.IONG and not self.EX1G):
            if(self.convManual):
                aManualConv1 = self.manual_convolution(aBorysov_Ion,aBorysov_Exc)
                aManualConv = aManualConv1
            else:
                aScipyConv1 = fftconvolve(aBorysov_Ion,aBorysov_Exc, mode=self.convMode)#, method='auto')
                aScipyConv = aScipyConv1
            print(f"sizes of input arrays for IONB={len(aBorysov_Ion)}, EX1B={len(aBorysov_Exc)}")
            print(f"sizes of convolutions for (IONB and EX1B and not IONG and not EX1G): IONBxEX1B={len(aManualConv1) if(self.convManual) else len(aScipyConv1)}")
        ### fill the model hist pdf
        aConv = aManualConv if(self.convManual) else aScipyConv
        xConv = np.linspace(start=self.dEmin,stop=self.dEmax,num=len(aConv))
        gConv = ROOT.TGraph(len(aConv),xConv, aConv)
        gConv.SetBit(ROOT.TGraph.kIsSortedX)
        for b in range(1,pdfs["hBorysov_Ion"].GetNbinsX()+1):
            # pdfs["hModel"].SetBinContent(b, aManualConv[b-1] if(self.convManual) else aScipyConv[b-1])
            xb = pdfs["hModel"].GetBinCenter(b)
            pdfs["hModel"].SetBinContent(b, gConv.Eval(xb+2*abs(self.dEmin)) )
        pdfs["hModel"].Scale(1./pdfs["hModel"].Integral())
        print(f'hModel={pdfs["hModel"].GetNbinsX()}, aConv={len(aConv)}')
        end = time.time()
        if(self.dotime):
            elapsed = end-start
            print(f"TIME of get_model_pdfs: {elapsed} [s]")
        return pdfs
    
    
    def get_cdfs(self,pdfs):
        start = time.time()
        cdfs = {}
        for name,pdf in pdfs.items():
            if(pdf==None): continue
            cdfs.update( {name : pdf.GetCumulative().Clone(name+"_cdf")} )
            cdfs[name].GetYaxis().SetTitle( cdfs[name].GetYaxis().GetTitle()+" (cumulative)" )
        end = time.time()
        if(self.dotime):
            elapsed = end-start
            print(f"TIME of get_cdfs: {elapsed} [s]")
        return cdfs
    
    
    def get_as_arrays(self,shapes,doScale=False):
        start = time.time()
        arrx  = []
        arrsy = {}
        rescale = (doScale and (self.IONB or self.EX1B or self.IONG or self.EX1G))
        for name,shape in shapes.items():
            if(shape==None): continue
            if(len(arrx)==0):
                for b in range(1,shape.GetNbinsX()+1):
                    x = shape.GetBinCenter(b)
                    if(rescale): x *= self.scale
                    arrx.append(x)
                arrx = np.array(arrx)
            arry = []
            for b in range(1,shape.GetNbinsX()+1):
                arry.append( shape.GetBinContent(b) )
            arry = np.array(arry)
            arrsy.update( {name : arry} )
        end = time.time()
        if(self.dotime):
            elapsed = end-start
            print(f"TIME of get_as_arrays: {elapsed} [s]")
        return arrx,arrsy
        
    def get_pdfs_from_arrays(self,arrx,arrsy,titles):
        start = time.time()
        pdfs = {}
        Nbins = len(arrx)
        dEbin = arrx[1]-arrx[0]
        dEmin = arrx[0]-dEbin/2.
        dEmax = arrx[-1]+dEbin/2.
        for name,arry in arrsy.items():
            h = ROOT.TH1D(name,titles,Nbins,dEmin,dEmax)
            for i in range(len(arrx)):
                xa = arrx[i]
                xh = h.GetBinCenter(i+1)
                if(abs(xa-xh)/xa>1e-6): print(f"xa={xa}, xh={xh}")
                y = h.SetBinContent(i+1, arry[i])
            pdfs.update({name:h})
            h.Scale(1./h.Integral())
            h.SetLineColor( ROOT.kRed )
            h.SetLineWidth( 1 )
        end = time.time()
        if(self.dotime):
            elapsed = end-start
            print(f"TIME of get_pdfs_from_arrays: {elapsed} [s]")
        return pdfs
        
    def set_all_shapes(self):
        start = time.time()
        ### get the basic pdfs
        self.cnt_pdfs = self.get_model_pdfs()
        self.sec_pdfs = self.get_secondaries_pdfs()
        ### make the cdfs from the pdfs
        self.cnt_cdfs = self.get_cdfs(self.cnt_pdfs)
        self.sec_cdfs = self.get_cdfs(self.sec_pdfs)
        ### get as arrays
        self.cnt_pdfs_arrx, self.cnt_pdfs_arrsy = self.get_as_arrays(self.cnt_pdfs, self.scale)
        self.sec_pdfs_arrx, self.sec_pdfs_arrsy = self.get_as_arrays(self.sec_pdfs,1)
        self.cnt_cdfs_arrx, self.cnt_cdfs_arrsy = self.get_as_arrays(self.cnt_cdfs, self.scale)
        self.sec_cdfs_arrx, self.sec_cdfs_arrsy = self.get_as_arrays(self.sec_cdfs,1)
        ### get as scaled arrays
        titles = self.cnt_pdfs["hModel"].GetTitle()+";"+self.cnt_pdfs["hModel"].GetXaxis().GetTitle()+";"+self.cnt_pdfs["hModel"].GetXaxis().GetTitle()
        self.cnt_pdfs_scaled = self.get_pdfs_from_arrays(self.cnt_pdfs_arrx,self.cnt_pdfs_arrsy,titles)
        self.cnt_cdfs_scaled = self.get_cdfs(self.cnt_pdfs_scaled)
        self.cnt_pdfs_scaled_arrx, self.cnt_pdfs_scaled_arrsy = self.get_as_arrays(self.cnt_pdfs_scaled, self.scale)
        self.cnt_cdfs_scaled_arrx, self.cnt_cdfs_scaled_arrsy = self.get_as_arrays(self.cnt_cdfs_scaled, self.scale)
        end = time.time()
        if(self.dotime):
            elapsed = end-start
            print(f"TIME of set_all_shapes: {elapsed} [s]")