import math
import array
import numpy as np
import ROOT
import bins
import units as U
import model

# ROOT.gROOT.SetBatch(1)
# ROOT.gStyle.SetOptFit(0)
# ROOT.gStyle.SetOptStat(0)
# ROOT.gStyle.SetPadBottomMargin(0.15)
# ROOT.gStyle.SetPadLeftMargin(0.13)
# ROOT.gStyle.SetPadRightMargin(0.15)

ROOT.gErrorIgnoreLevel = ROOT.kError
# ROOT.gErrorIgnoreLevel = ROOT.kWarning

class ToyMC:
    def __init__(self,E,dx,model):
        self.E     = E
        self.dx    = dx
        self.point = ("%.1f" % (E*U.eV2MeV))+"MeV_"+("%.4f" % (dx*U.cm2um))+"um"
        self.model = model
        self.rnd   = ROOT.TRandom() ## random engine
        print(f"Generating for slice {self.point}")
    
    def Generate(self,Nsteps):
        BEBL = self.model.BEBL
        IONB = self.model.IONB
        EX1B = self.model.EX1B
        IONG = self.model.IONG
        EX1G = self.model.EX1G
        
        histos = {}
        histos.update(           { "hTotal":        ROOT.TH1D("hTotal",       "Toy Data vs FULL Model for "+self.point+";#DeltaE [eV];Steps",self.model.Nbins,self.model.dEmin,self.model.dEmax) } )
        if(IONB): histos.update( { "hIon_non_gaus": ROOT.TH1D("hIon_non_gaus","Toy Data vs IONB Model for "+self.point+";#DeltaE [eV];Steps",self.model.Nbins,self.model.dEmin,self.model.dEmax) } )
        if(EX1B): histos.update( { "hExc_non_gaus": ROOT.TH1D("hExc_non_gaus","Toy Data vs EX1B Model for "+self.point+";#DeltaE [eV];Steps",self.model.Nbins,self.model.dEmin,self.model.dEmax) } )
        if(IONG): histos.update( { "hIon_gaus":     ROOT.TH1D("hIon_gaus",    "Toy Data vs IONG Model for "+self.point+";#DeltaE [eV];Steps",self.model.Nbins,self.model.dEmin,self.model.dEmax) } )
        if(EX1G): histos.update( { "hExc_gaus":     ROOT.TH1D("hExc_gaus",    "Toy Data vs EX1G Model for "+self.point+";#DeltaE [eV];Steps",self.model.Nbins,self.model.dEmin,self.model.dEmax) } )
        for hname,hist in histos.items():
            hist.SetLineColor(ROOT.kBlack)
            hist.SetMarkerColor(ROOT.kBlack)
            hist.SetMarkerStyle(20)
            hist.SetMarkerSize(0.6)

        ### run...
        for  i in range(Nsteps):
            ### monitor
            if(i>0 and i%1000000==0): print(f"Processed {i} events")
            ### initialize
            eloss_Exc_non_gaus = 0
            eloss_Ion_non_gaus = 0
            eloss_Exc_gaus     = 0
            eloss_Ion_gaus     = 0
            EX1BCOND = False
            IONBCOND = False
            EX1GCOND = False
            IONGCOND = False
    
            ### if average loss is smaller than 10 eV, take the averge loss
            if(BEBL):
                hTotal.Fill(meanLoss)
                continue
    
            ### excitation non-gauss
            if(EX1B):
                p = self.rnd.Poisson(self.model.n1)
                if(p>0): eloss_Exc_non_gaus = self.model.e1*((p + 1.) - 2.*self.rnd.Uniform())
                EX1BCOND = (p>0 and eloss_Exc_non_gaus>0)
                if(EX1BCOND): histos["hExc_non_gaus"].Fill( eloss_Exc_non_gaus )
    
            ### ionization non-gauss
            if(IONB):
                nnb = self.rnd.Poisson(self.model.p3)
                if(nnb>0):
                    for k in range(nnb): eloss_Ion_non_gaus += self.model.w3/(1.-self.model.w*self.rnd.Uniform()) ## actually this cannot be be smaller than w3
                IONBCOND = (nnb>0 and eloss_Ion_non_gaus>0)
                if(IONBCOND): histos["hIon_non_gaus"].Fill( eloss_Ion_non_gaus ) ## actually this cannot be smaller than w3
    
            ### excitation gauss
            if(EX1G):
                eloss_Exc_gaus = self.rnd.Gaus(self.model.ex1_mean,self.model.ex1_sigma)
                EX1GCOND = (eloss_Exc_gaus>0 and eloss_Exc_gaus<2*self.model.ex1_mean)
                if(EX1GCOND): histos["hExc_gaus"].Fill( eloss_Exc_gaus )
    
            ### ionization gauss
            if(IONG):
                eloss_Ion_gaus = self.rnd.Gaus(self.model.ion_mean,self.model.ion_sigma)
                IONGCOND = (eloss_Ion_gaus>0 and eloss_Ion_gaus<2*self.model.ion_mean)
                if(IONGCOND): histos["hIon_gaus"].Fill( eloss_Ion_gaus )

            ### sum it up
            Eloss = 0
            if(IONB and IONG and EX1G): ## borysov ion + gaus ion + gaus exc
                if(IONBCOND and EX1GCOND and IONGCOND):
                    Eloss = eloss_Ion_non_gaus+eloss_Exc_gaus+eloss_Ion_gaus
            if(IONB and EX1B and IONG): ## borysov ion + borysov exc + gaus ion
                if(IONBCOND and EX1BCOND and IONGCOND):
                    Eloss = eloss_Ion_non_gaus+eloss_Exc_non_gaus+eloss_Ion_gaus
            if(IONB and EX1B and not IONG and not EX1G):
                if(IONBCOND and EX1BCOND):  ## borysov ion + borysov exc
                    Eloss = eloss_Ion_non_gaus+eloss_Exc_non_gaus
            if(Eloss>0): histos["hTotal"].Fill(Eloss)
        return histos
