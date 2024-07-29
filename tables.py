import array
import math
import numpy as np
import ROOT
import units as U
import constants as C
import bins


class Tables:
    def __init__(self,fname_dEdx,fname_range="",fname_invrng=""):
        ### parameters
        self.linLossLimit     = 0.01
        self.massRatio        = 1.
        self.charge2ratio     = 1.
        self.biasFactor       = 1.
        self.fFactor          = self.charge2ratio*self.biasFactor
        self.reduceFactor     = 1.0/(self.fFactor*self.massRatio)
        self.fRangeEnergy     = 0.
        self.minKinEnergy     = 0.1*U.keV2eV
        ### dE/dx objects
        self.dEdxTable           = {"E":None,"dEdx":None}
        self.dEdxTable_eV_per_cm = {"E":None,"dEdx":None}
        self.rangeTable          = {"E":None,"R":None}
        self.invrngTable         = {"R":None,"E":None}
        ### set the dE/dx objects
        self.setdEdxTableFromFile(fname_dEdx)
        self.dEdxGraph           = self.getGraph(self.dEdxTable["E"],self.dEdxTable["dEdx"],"E [MeV]","dE/dx [MeV/mm]")
        self.dEdxGraph_eV_per_cm = self.getGraph(self.dEdxTable_eV_per_cm["E"],self.dEdxTable_eV_per_cm["dEdx"],"E [eV]","dE/dx [eV/cm]")
        ### set the range objects
        if(fname_range==""):  self.BuildRangeTable()
        else:                 self.setRangeTableFromFile(fname_range)
        if(fname_invrng==""): self.BuildInverseRangeTable()
        else:                 self.setInvrngTableFromFile(fname_invrng)
        self.rangeGraph  = self.getGraph(self.rangeTable["E"],self.rangeTable["R"],"E [MeV]","Range [?]")
        self.invrngGraph = self.getGraph(self.invrngTable["R"],self.invrngTable["E"],"Range [?]", "E [MeV]")
    
    def getGraph(self,X,Y,xtitle="",ytitle=""):
        g = ROOT.TGraph(len(X),X,Y)
        g.SetBit(ROOT.TGraph.kIsSortedX)
        g.SetLineColor(ROOT.kBlue)
        g.GetXaxis().SetTitle(xtitle)
        g.GetYaxis().SetTitle(ytitle)
        return g

    def setdEdxTableFromFile(self,fname):
        hname = fname
        hname = hname.split(".")[0]
        arr_E               = array.array( 'd' )
        arr_dEdx            = array.array( 'd' )
        arr_E_eV            = array.array( 'd' )
        arr_dEdx_eV_per_cm  = array.array( 'd' )
        with open(fname) as f:
            for line in f:
                if("#" in line): continue
                line = line.replace("\n","")
                words = line.split("  ")
                arr_E.append( float(words[0]) )
                arr_dEdx.append( float(words[1]) )    
                arr_E_eV.append( float(words[0]) * U.MeV2eV )
                arr_dEdx_eV_per_cm.append( float(words[1]) * (U.MeV2eV/(U.mm2cm)) )
        npts = len(arr_E)
        print(f"Read {npts} points from file {fname}")
        ### set the dE/dx table
        self.dEdxTable["E"]    = arr_E
        self.dEdxTable["dEdx"] = arr_dEdx
        self.dEdxTable_eV_per_cm["E"]    = arr_E_eV
        self.dEdxTable_eV_per_cm["dEdx"] = arr_dEdx_eV_per_cm
    
    def setRangeTableFromFile(self,fname):
        hname = fname
        hname = hname.split(".")[0]
        arr_E = array.array( 'd' )
        arr_R = array.array( 'd' )
        with open(fname) as f:
            for line in f:
                if("#" in line): continue
                line = line.replace("\n","")
                words = line.split("  ")
                arr_E.append( float(words[0]) )
                arr_R.append( float(words[1]) )    
        npts = len(arr_E)
        print(f"Read {npts} points from file {fname}")
        ### set the range table
        self.rangeTable["E"] = arr_E
        self.rangeTable["R"] = arr_R
        
    def setInvrngTableFromFile(self,fname):
        hname = fname
        hname = hname.split(".")[0]
        arr_R = array.array( 'd' )
        arr_E = array.array( 'd' )
        with open(fname) as f:
            for line in f:
                if("#" in line): continue
                line = line.replace("\n","")
                words = line.split("  ")
                arr_R.append( float(words[0]) )
                arr_E.append( float(words[1]) )    
        npts = len(arr_R)
        print(f"Read {npts} points from file {fname}")
        ### set the range table
        self.invrngTable["R"] = arr_R
        self.invrngTable["E"] = arr_E
    
    def BuildRangeTable(self):
        arr_E = array.array( 'd' )
        arr_R = array.array( 'd' )
        n = 100
        Del = 1./float(n)
        bin0 = 0
        npoints = len(self.dEdxTable["E"])
        elow  = self.dEdxTable["E"][0]
        ehigh = self.dEdxTable["E"][-1]
        dedx1 = self.dEdxTable["dEdx"][0]
        ### protection for specific cases dedx=0
        if(dedx1==0.):
            for k in range(1,npoints):
                bin0 += 1
                elow  = self.dEdxTable["E"][k]
                dedx1 = self.dEdxTable["dEdx"][k]
                if(dedx1>0.): break
            npoints -= bin0
        ### initialisation of a new vector
        if(npoints<3): npoints = 3
        ### assumed dedx proportional to beta
        energy1 = self.dEdxTable["E"][0]
        Range   = 2.*energy1/dedx1
        ### fill the range table
        arr_E.append( energy1 ) 
        arr_R.append( Range ) 
        for j in range(1,npoints):
            energy2 = self.dEdxTable["E"][j]
            dE      = (energy2 - energy1) * Del
            energy  = energy2 + dE*0.5
            Sum = 0.
            idx = j - 1
            for k in range(n):
                energy -= dE
                dedx1 = self.dEdxGraph.Eval(energy)
                if(dedx1>0.): Sum += dE/dedx1
            Range += Sum
            arr_E.append( energy2 )
            arr_R.append( Range )
            energy1 = energy2
        ### make the table
        self.rangeTable["E"] = arr_E
        self.rangeTable["R"] = arr_R

    def BuildInverseRangeTable(self):
        ### Build inverse range table from the energy loss table
        arr_R = array.array( 'd' )
        arr_E = array.array( 'd' )
        npoints = len(self.rangeTable["E"])
        for j in range(npoints):
            E = self.rangeTable["E"][j]
            R = self.rangeTable["R"][j]
            arr_R.append(R)
            arr_E.append(E)
        self.invrngTable["R"] = arr_R
        self.invrngTable["E"] = arr_E

    def getFrange(self,E):
        fRange = 0
        if(not self.fRangeEnergy==E):
            fRange = self.reduceFactor*self.rangeGraph.Eval(E*U.eV2MeV)
            if(fRange<0.):
                fRange = 0.
            elif(E<self.minKinEnergy):
                fRange *= math.sqrt(E/self.minKinEnergy)
        return fRange

    def ScaledKinEnergyForLoss(self,R):
        Rmin = self.invrngTable["R"][0]
        E = 0.
        if(R>=Rmin): E = self.invrngGraph.Eval(R)
        elif(R>0.):
            xx = R/Rmin
            E = self.minKinEnergy*xx*xx
        return E*U.MeV2eV

    def getMeanLoss(self,E,x):
        meanLoss = x*self.dEdxGraph_eV_per_cm.Eval(E) # eV
        fRange = self.getFrange(E)
        if(meanLoss > E*self.linLossLimit):
            xx = (fRange-x*U.cm2mm)/self.reduceFactor
            dE = E - self.ScaledKinEnergyForLoss(xx)/self.massRatio
            if(dE>0.0): meanLoss = dE
        return meanLoss
    
###############################################
###############################################
###############################################

if __name__ == "__main__":
    # fname_dEdx   = "inputs/eloss_p_si.txt"
    fname_dEdx   = "inputs/dEdx_p_si.txt"
    fname_range  = "inputs/range_p_si.txt"
    fname_invrng = "inputs/invrng_p_si.txt"
    tables = Tables(fname_dEdx,fname_range,fname_invrng)
    
    '''
    kinetic energy: 0.470612 MeV
    length:         5.34713 um
    averageLoss:    0.448115 MeV
    siga:           0.0098689414322724132 MeV
    '''
    E = 0.470612*U.MeV2eV
    x = 5.347130*U.um2cm
    averageLoss = 0.448115 #MeV
    modifiedMeanLoss = tables.getMeanLoss(E,x) # eV
    print(f"Kinetic energy={E} [eV]")
    print(f"Step length={x} [cm]")
    print(f"GEAT4 meanLoss={averageLoss} [MeV]")
    print(f"Graph meanLoss={tables.dEdxGraph_eV_per_cm.Eval(E)*x*U.eV2MeV} [MeV]")
    print(f"Modified meanLoss={modifiedMeanLoss*U.eV2MeV} [MeV]")
    
    
    # h = ROOT.TH2D("SMALL_h_dL_vs_E",";E [MeV];#DeltaL [#mum];Steps", len(bins.Ebins_small)-1,array.array("d",bins.Ebins_small), len(bins.dLbins_small)-1,array.array("d",bins.dLbins_small))
    #
    # for bx in range(1,h.GetNbinsX()+1):
    #     for by in range(1,h.GetNbinsY()+1):
    #         E = h.GetXaxis().GetBinCenter(bx)*U.MeV2eV
    #         L = h.GetYaxis().GetBinCenter(by)*U.um2cm
    #         meanLoss = L*tables.dEdxGraph_eV_per_cm.Eval(E) * U.eV2MeV
    #         modifiedMeanLoss = tables.getMeanLoss(E,L) * U.eV2MeV
    #         print(f"[{bx},{by}]: E={E*U.eV2MeV:.3g} [MeV], L={L*U.cm2um:.3g} [um], meanLoss={meanLoss:.3g} [MeV] --> modLoss={modifiedMeanLoss:.3g} [MeV]........isSame? {(abs(modifiedMeanLoss-meanLoss)/meanLoss<0.001)}")
    
    
    