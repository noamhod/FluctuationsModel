import math
import array
import numpy as np
import ROOT
import units as U
import particle as prt
from scipy.fft import fft, fftfreq, rfft, irfft
from scipy.special import sici, exp1
from scipy.signal import convolve, fftconvolve
from scipy.interpolate import interp1d
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
        xx += math.pow(par[0],ii) / ROOT.TMath.Factorial(ii)
    xx *= 0.5*ROOT.TMath.Exp(-par[0])/par[1] ### TODO: this does NOT include the case of zero-loss and I deal with it later in get_pdf()
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
        self.doprint   = False  
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
        self.primprt   = self.param["primprt"]   if("primprt"   in self.param) else prt.Particle(name="NONE",meV=-999,mamu=-999,chrg=-999,spin=-999,lepn=-999,magm=-999)
        self.minLoss   = self.param["minLoss"]   if("minLoss"   in self.param) else -1
        self.meanLoss  = self.param["meanLoss"]  if("meanLoss"  in self.param) else -1
        self.Tcut      = self.param["Tcut"]      if("Tcut"      in self.param) else -1
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
        self.thk_mean  = self.param["thk_mean"]  if("thk_mean"  in self.param) else -1
        self.thk_sigma = self.param["thk_sigma"] if("thk_sigma" in self.param) else -1
        self.thk_neff  = self.param["thk_neff"]  if("thk_neff"  in self.param) else -1
        ### set parameters
        self.par_bethebloch_min  = [self.meanLoss]
        self.par_zero_loss       = [0]
        self.par_borysov_sec     = [self.EkinMin, self.EkinMax]
        self.par_borysov_ion     = [self.w3, self.w, self.p3]
        self.par_borysov_exc     = [self.n1, self.e1]
        self.par_gauss_ion       = [self.ion_mean, self.ion_sigma]
        self.par_gauss_exc       = [self.ex1_mean, self.ex1_sigma]
        self.par_gauss_thk       = [self.thk_mean, self.thk_sigma]
        self.par_gamma_thk       = [self.meanLoss, self.thk_neff]
        
        self.NptsTF1    = 100000
        
        ### continuous, unscaled dE axis
        self.dEmin      = -1
        self.dEmax      = -1
        self.Nbins      = -1
        ### continuous, scaled dE axis
        self.NbinsScl   = -1
        self.dEminScl   = -1
        self.dEmaxScl   = -1
        ### secondaries, unscaled dE axis (always the case)
        self.dEminSec   = -1
        self.dEmaxSec   = -1
        self.NbinsSec   = -1
        ### relevant for everything
        self.doLogx     = False
        
        ### intialize everything else
        self.set_flags()
        self.validate_pars()
        self.dE_binning()
        ### for the frequencies spacing
        self.min_N_t_bins = 500000
        self.N_t_bins  = -1
        self.tmin      = -1
        self.tmax      = -1
        self.psiRe     = None
        self.psiIm     = None
        self.plotPsi   = True
        self.fSampling = -1
        self.TSampling = -1
        

    def TimeIt(self,start,end,name):
        if(self.dotime):
            elapsed = end-start
            print(f"TIME of {name} is: {elapsed} [s]")
    
    ### get the flags from the build string
    def set_flags(self):
        self.BEBL = (self.meanLoss<self.minLoss and "BEBL" in self.build)
        self.TGAU = ("THK.GAUSS" in self.build)
        self.TGAM = ("THK.GAMMA" in self.build)
        self.IONB = ("ION.B" in self.build)
        self.EX1B = ("EX1.B" in self.build)
        self.IONG = ("ION.G" in self.build)
        self.EX1G = ("EX1.G" in self.build)
        self.SECB = ("SEC.B" in self.build)
        if(self.doprint): print(f"SECB={self.SECB}, BEBL={self.BEBL}, IONB={self.IONB}, EX1B={self.EX1B}, IONG={self.IONG}, EX1G={self.EX1G}")

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
            self.dEminSec  = 0.5*self.Tcut #10
            self.dEmaxSec  = 5000000.1
            self.NbinsSec  = 100000
        if(self.BEBL):
            self.dEmin     = 0
            self.dEmax     = 11
            self.Nbins     = 11
        if(self.TGAU or self.TGAM):
            self.dEmin     = 1000
            self.dEmax     = 10000000+self.dEmin
            self.Nbins     = 50000
        if(self.IONB and self.EX1B and not self.IONG and not self.EX1G): ## Borysov only, no Gauss
            self.dEmin     = 0 #0.1 #0.05
            self.dEmax     = 10000 #10000.1 #10000.05
            self.Nbins     = 10000
        if(self.IONB and not self.EX1B and self.IONG and self.EX1G): ## no Borysov Exc
            self.dEmin     = 0 #0.1 #10
            self.dEmax     = 1000000 #1000000.1 #1000010
            self.Nbins     = 50000
        if(self.IONB and (self.IONG and not self.EX1G) or (self.EX1G and not self.IONG)): ## only one Gauss
            self.dEmin     = 0 #0.1 #10
            self.dEmax     = 100000 #100000.1 #100010
            self.Nbins     = 10000
        self.set_scaled_xaxis_binning() ### only for continuous, non-BEBL,non-Thick
        self.doLogx = True if(self.dEmin>0) else False
        if(self.doprint): print(f"dEmin={self.dEmin}, dEmax={self.dEmax}, Nbins={self.Nbins}")
    
    def set_fft_sampling_pars(self,N_t_bins,frac):
        if(not self.IONB): return
        start = time.time()
        self.N_t_bins = N_t_bins
        self.tmin = -500
        self.tmax = +500
        ### first find the t range
        trange, psiRe, psiIm = self.scipy_psi_of_t()
        psiRe_mean = np.mean(psiRe)
        psiIm_mean = np.mean(psiIm)
        psiRe_stdv = np.std(psiRe)
        psiIm_stdv = np.std(psiIm)
        for k,t in enumerate(trange):
            if((psiRe[k]>psiRe_mean+psiRe_stdv*frac) or (psiRe[k]<psiRe_mean-psiRe_stdv*frac) or (psiIm[k]>psiIm_mean+psiIm_stdv*frac) or (psiIm[k]<psiIm_mean-psiIm_stdv*frac)):
                self.tmin = t
                self.tmax = abs(t)
                if(k>0):
                    self.N_t_bins = (N_t_bins-2*k)
                    if(self.N_t_bins<self.min_N_t_bins): self.N_t_bins = self.min_N_t_bins
                if(self.doprint): print(f"Changing to tmin=-500-->{self.tmin}, tmax=+500-->{self.tmax}, N_t_bins={N_t_bins}-->{self.N_t_bins}")
                break
        self.fSampling = (2*np.pi)*(self.N_t_bins/(self.tmax-self.tmin))
        self.TSampling = 1./self.fSampling
        if(self.doprint): print(f"tmin={self.tmin}, tmax={self.tmax}, N_t_bins={self.N_t_bins}, TSampling={self.TSampling}")
        end = time.time()
        self.TimeIt(start,end,"set_fft_sampling_pars")
    
    def set_fft_sampling_pars_rotem(self, N_t_bins, frac):
        if(not self.IONB): return
        start = time.time()
        self.tmin = -500
        self.tmax = +500
        self.N_t_bins = N_t_bins
        # first find the trange
        trange, psiRe, psiIm = self.scipy_psi_of_t()
        
        psiRe_mean = np.mean(psiRe)
        psiIm_mean = np.mean(psiIm)
        psiRe_stdv = np.std(psiRe)
        psiIm_stdv = np.std(psiIm)
        
        psiRe_plus_stdv  = psiRe_mean+psiRe_stdv*frac
        psiRe_minus_stdv = psiRe_mean-psiRe_stdv*frac
        psiIm_plus_stdv  = psiIm_mean+psiIm_stdv*frac
        psiIm_minus_stdv = psiIm_mean-psiIm_stdv*frac
        
        locs_psiRe_plus_stdv  = psiRe > psiRe_plus_stdv
        locs_psiRe_minus_stdv = psiRe < psiRe_minus_stdv
        locs_psiIm_plus_stdv  = psiIm > psiIm_plus_stdv
        locs_psiIm_minus_stdv = psiIm < psiIm_minus_stdv
        
        locs = locs_psiRe_plus_stdv | locs_psiRe_minus_stdv | locs_psiIm_plus_stdv | locs_psiIm_minus_stdv
        first_true_loc = int(np.argmax(locs))  # first True value in 'locs'
        self.tmin = trange[first_true_loc]
        self.tmax = abs(trange[first_true_loc])
        if(first_true_loc>0):
            self.N_t_bins = int(N_t_bins-2*first_true_loc)
            if(self.N_t_bins<self.min_N_t_bins):
                self.N_t_bins = self.min_N_t_bins
        if(self.doprint): print(f"Changing to tmin=-500-->{self.tmin}, tmax=+500-->{self.tmax}, N_t_bins={N_t_bins}-->{self.N_t_bins}")
        self.fSampling = (2*np.pi)*(self.N_t_bins/(self.tmax-self.tmin))
        self.TSampling = 1./self.fSampling
        if(self.doprint): print(f"tmin={self.tmin}, tmax={self.tmax}, N_t_bins={self.N_t_bins}, TSampling={self.TSampling}")
        end = time.time()
        self.TimeIt(start,end,"set_fft_sampling_pars_rotem")
    
    def scipy_psi_of_t(self):
        if(not self.IONB or self.N_t_bins<0): return
        ## par[0] = w3
        ## par[1] = w
        ## par[2] = p3 (lambda)
        ## par[3] = 0/1 Re/Im
        start = time.time()
        par = self.par_borysov_ion
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
        end = time.time()
        self.TimeIt(start,end,"scipy_psi_of_t")
        return t, psi_re, psi_im
    
    def scipy_psi_of_t_as_h(self,name):
        if(self.psiRe is None or self.psiIm is None):
            print("psiRe/psiIm are None. Quit.")
            quit()
        start = time.time()
        h_re = ROOT.TH1D("h_re_"+name,"Borysov Re[#psi(t)];t [1/eV];Re[#psi(t)]",self.N_t_bins,self.tmin,self.tmax)
        h_im = ROOT.TH1D("h_im_"+name,"Borysov Im[#psi(t)];t [1/eV];Im[#psi(t)]",self.N_t_bins,self.tmin,self.tmax)
        for bb in range(1,h_re.GetNbinsX()+1):
            x = h_re.GetBinCenter(bb)
            y_re = self.psiRe[bb-1]
            y_im = self.psiIm[bb-1]
            h_re.SetBinContent(bb,y_re)
            h_im.SetBinContent(bb,y_im)
        h_re.SetLineColor( ROOT.kRed )
        h_im.SetLineColor( ROOT.kBlue )
        h_re.SetLineWidth( 1 )
        h_im.SetLineWidth( 1 )
        end = time.time()
        self.TimeIt(start,end,"scipy_psi_of_t_as_h")
        return h_re,h_im

    def borysov_ionization(self,par):
        start = time.time()
        ### get psi(t)
        t, self.psiRe, self.psiIm = self.scipy_psi_of_t()
        ### The FFT
        y = self.psiRe + 1.j*self.psiIm
        yf = fft(y)
        xf = fftfreq(self.N_t_bins, self.TSampling)[:self.N_t_bins//2] ## remove the last M elements, where M=floor(N/2)
        ya = np.abs(yf[0:self.N_t_bins//2])*(2/self.N_t_bins) ### TODO: this does NOT include the case of zero-loss and I deal with it later in get_pdf()
        ### Get the integral as graph
        gFFT = ROOT.TGraph(len(xf),xf,ya)
        gFFT.SetLineColor( ROOT.kRed )
        gFFT.SetBit(ROOT.TGraph.kIsSortedX)
        end = time.time()
        self.TimeIt(start,end,"borysov_ionization")
        return gFFT

    def get_pdf(self,name,pdfname,par):
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
                for i in range(len(par)):
                    if(f is not None): f.SetParameter(i,par[i])
                    else:
                        print(f"TF1 object is None for pdfname={pdfname}")
                        quit()
                f.SetNpx(self.NptsTF1)
                f.SetLineColor(ROOT.kRed)
            for bb in range(1,h.GetNbinsX()+1):
                x = h.GetBinCenter(bb)
                y = g.Eval(x) if(isBorysovIon) else f.Eval(x)
                h.SetBinContent(bb,y)
            #######################################
            ### TODO: implement the thick cases ###
            #######################################
        
        ### renormalize the two parts of the pdf in non-gauss ion/exc:
        if(pdfname=="borysov_ionization" or pdfname=="borysov_excitation"):
            poisson_lambda = par[2] if(pdfname=="borysov_ionization") else par[0]
            norm_zero = math.exp(-poisson_lambda)
            norm_non0 = 1.-norm_zero
            h.Scale(norm_non0/h.Integral())
            h.AddBinContent(1,norm_zero) ## then add the "zero-loss" contribution to the first bin - assuming that the histos start at dE=0!
        else:
            h.Scale(1./h.Integral())
        
        h.SetLineColor( ROOT.kRed )
        h.SetLineWidth( 1 )
        end = time.time()
        self.TimeIt(start,end,f"get_pdf({name})")
        return h
    
    def convolution_fft(self,X,A,K):
        start = time.time()
        N = len(K)
        if(len(A)!=N):
            print("Arrays size does not match. Quit")
            quit()
        Y = fftconvolve(A,K)
        DX = X[1]-X[0]
        ## the extended x range where the convolution is defiend in:
        X_extended = np.linspace(X[0], X[-1]+DX*(N-1), 2*N-1)
        # now interpolate Y to the original X range
        interp_func = interp1d(X_extended,Y,kind='linear') 
        Y_interpolated = interp_func(X)
        end = time.time()
        self.TimeIt(start,end,"convolution_fft")
        return Y_interpolated
    
    def get_continuous_pdfs(self):
        start = time.time()
        ### get pdfs
        pdfs = {}
        pdfs.update({"hModel":        None})
        pdfs.update({"hBorysov_Ion":  None})
        pdfs.update({"hBorysov_Exc":  None})
        pdfs.update({"hTrncGaus_Ion": None})
        pdfs.update({"hTrncGaus_Exc": None})
        pdfs.update({"hBEBL_Thn":     None})
        pdfs.update({"hTrncGaus_Thk": None})
        pdfs.update({"hGamma_Thk":    None})
        pdfs["hModel"] = ROOT.TH1D("hModel","",self.Nbins,self.dEmin,self.dEmax)
        pdfs["hModel"].SetLineColor(ROOT.kRed)
        if(self.BEBL):
            pdfs["hBEBL_Thn"] = self.get_pdf("bethebloch_min_model", "bethebloch_min_model", self.par_bethebloch_min)
        if(self.IONB and self.EX1B and self.IONG):
            pdfs["hBorysov_Ion"]  = self.get_pdf("borysov_ion_model", "borysov_ionization", self.par_borysov_ion)
            pdfs["hBorysov_Exc"]  = self.get_pdf("borysov_exc_model", "borysov_excitation", self.par_borysov_exc)
            pdfs["hTrncGaus_Ion"] = self.get_pdf("gauss_ion_model",   "truncated_gaus",     self.par_gauss_ion)
        if(self.IONB and self.IONG and self.EX1G):
            pdfs["hBorysov_Ion"]  = self.get_pdf("borysov_ion_model", "borysov_ionization", self.par_borysov_ion)
            pdfs["hTrncGaus_Ion"] = self.get_pdf("gauss_ion_model",   "truncated_gaus",     self.par_gauss_ion)
            pdfs["hTrncGaus_Exc"] = self.get_pdf("gauss_exc_model",   "truncated_gaus",     self.par_gauss_exc)
        if(self.IONB and self.EX1B and not self.IONG and not self.EX1G):
            pdfs["hBorysov_Ion"] = self.get_pdf("borysov_ion_model", "borysov_ionization",  self.par_borysov_ion)
            pdfs["hBorysov_Exc"] = self.get_pdf("borysov_exc_model", "borysov_excitation",  self.par_borysov_exc)
        if(self.TGAU):
            pdfs["hTrncGaus_Thk"]   = self.get_pdf("gauss_thk_model",   "truncated_gaus",   self.par_gauss_thk)
        if(self.TGAM):
            pdfs["hGamma_Thk"]      = self.get_pdf("gamma_thk_model",   "gamma_gaus",       self.par_gamma_thk)
        end = time.time()
        self.TimeIt(start,end,"get_continuous_pdfs")
        return pdfs
        
    def get_secondaries_pdfs(self):
        start = time.time()
        ### get pdfs
        pdfs = {}
        pdfs.update({"hBorysov_Sec":  None})
        if(self.SECB):
            pdfs["hBorysov_Sec"] = self.get_pdf("borysov_sec_model", "borysov_secondaries", self.par_borysov_sec)
        end = time.time()
        self.TimeIt(start,end,"get_secondaries_pdfs")
        return pdfs
    
    def get_model_pdfs(self):
        start = time.time()
        ### get the pdfs of the continuous part
        pdfs = self.get_continuous_pdfs()
        ### if meanLoss is too small return the meanLoss as single bin PDF
        if(self.BEBL):
            for b in range(1,pdfs["hBEBL_Thn"].GetNbinsX()+1):
                pdfs["hModel"].SetBinContent(b, pdfs["hBEBL_Thn"].GetBinContent(b) )
            return pdfs
            
        if(self.TGAU):
            for b in range(1,pdfs["hTrncGaus_Thk"].GetNbinsX()+1):
                pdfs["hModel"].SetBinContent(b, pdfs["hTrncGaus_Thk"].GetBinContent(b) )
            return pdfs

        if(self.TGAM):
            for b in range(1,pdfs["hGamma_Thk"].GetNbinsX()+1):
                pdfs["hModel"].SetBinContent(b, pdfs["hGamma_Thk"].GetBinContent(b) )
            return pdfs
            
        ### Otherwise, constructe the continuous model 
        aBorysov_Ion  = np.zeros( pdfs["hBorysov_Ion"].GetNbinsX() )
        aBorysov_Exc  = np.zeros( pdfs["hBorysov_Ion"].GetNbinsX() )
        aTrncGaus_Ion = np.zeros( pdfs["hBorysov_Ion"].GetNbinsX() )
        aTrncGaus_Exc = np.zeros( pdfs["hBorysov_Ion"].GetNbinsX() )
        aX            = np.zeros( pdfs["hBorysov_Ion"].GetNbinsX() )
        if(not self.BEBL and not self.TGAU and not self.TGAM):
            for b in range(1,pdfs["hBorysov_Ion"].GetNbinsX()+1):
                if(pdfs["hBorysov_Ion"]  is not None): aX[b-1]            = pdfs["hBorysov_Ion"].GetBinCenter(b)
                if(pdfs["hBorysov_Ion"]  is not None): aBorysov_Ion[b-1]  = pdfs["hBorysov_Ion"].GetBinContent(b)
                if(pdfs["hBorysov_Exc"]  is not None): aBorysov_Exc[b-1]  = pdfs["hBorysov_Exc"].GetBinContent(b)
                if(pdfs["hTrncGaus_Ion"] is not None): aTrncGaus_Ion[b-1] = pdfs["hTrncGaus_Ion"].GetBinContent(b)
                if(pdfs["hTrncGaus_Exc"] is not None): aTrncGaus_Exc[b-1] = pdfs["hTrncGaus_Exc"].GetBinContent(b)
        aFFTConv = None
        if(self.IONB and self.EX1B and self.IONG):
            aFFTConv1 = self.convolution_fft(aX,aBorysov_Ion,aBorysov_Exc)
            aFFTConv2 = self.convolution_fft(aX,aTrncGaus_Ion,aFFTConv1)
            aFFTConv = aFFTConv2
        if(self.IONB and self.IONG and self.EX1G):
            aFFTConv1 = self.convolution_fft(aX,aBorysov_Ion,aTrncGaus_Ion)
            aFFTConv2 = self.convolution_fft(aX,aTrncGaus_Exc,aFFTConv1)
            aFFTConv = aFFTConv2
        if(self.IONB and self.EX1B and not self.IONG and not self.EX1G):
            aFFTConv1 = self.convolution_fft(aX,aBorysov_Ion,aBorysov_Exc)
            aFFTConv = aFFTConv1
        ### fill the model hist pdf
        aConv = aFFTConv
        xConv = np.linspace(start=self.dEmin,stop=self.dEmax,num=len(aConv))
        for b in range(1,pdfs["hBorysov_Ion"].GetNbinsX()+1):
            pdfs["hModel"].SetBinContent(b,aConv[b-1])
        # pdfs["hModel"].Scale(1./pdfs["hModel"].Integral())
        end = time.time()
        self.TimeIt(start,end,"get_model_pdfs")
        return pdfs
    
    def get_cdfs(self,pdfs):
        start = time.time()
        cdfs = {}
        for name,pdf in pdfs.items():
            if(pdf==None):
                cdfs.update( {name : None} )
                continue
            cdfs.update( {name : pdf.GetCumulative().Clone(name+"_cdf")} )
            cdfs[name].GetYaxis().SetTitle( cdfs[name].GetYaxis().GetTitle()+" (cumulative)" )
        end = time.time()
        self.TimeIt(start,end,"get_cdfs")
        return cdfs
    
    ### for defining histos as in the scaled model
    def set_scaled_xaxis_binning(self):
        rescale = (self.IONB or self.EX1B or self.IONG or self.EX1G)
        self.NbinsScl = self.Nbins
        self.dEminScl = self.dEmin*self.scale if(rescale) else self.dEmin
        self.dEmaxScl = self.dEmax*self.scale if(rescale) else self.dEmax
    
    def get_as_arrays(self,shapes,doScale=False):
        start = time.time()
        arrx  = []
        arrsy = {}
        rescale = (doScale and (self.IONB or self.EX1B or self.IONG or self.EX1G))
        for name,shape in shapes.items():
            if(shape==None): continue
            if(len(arrx)==0):
                arrx = np.zeros( shape.GetNbinsX() )
                for b in range(1,shape.GetNbinsX()+1):
                    x = shape.GetBinCenter(b)
                    if(rescale): x *= self.scale
                    arrx[b-1] = x
            arry = np.zeros(shape.GetNbinsX())
            for b in range(1,shape.GetNbinsX()+1):
                arry[b-1] = shape.GetBinContent(b)
            arrsy.update( {name : arry} )
        end = time.time()
        self.TimeIt(start,end,"get_as_arrays")
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
                if(abs(xa-xh)/xa>1e-6): print(f"WARNING: xa={xa}, xh={xh}")
                y = h.SetBinContent(i+1, arry[i])
            pdfs.update({name:h})
            # h.Scale(1./h.Integral())
            h.SetLineColor( ROOT.kRed )
            h.SetLineWidth( 1 )
        end = time.time()
        self.TimeIt(start,end,"get_pdfs_from_arrays")
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
        self.TimeIt(start,end,"set_all_shapes")