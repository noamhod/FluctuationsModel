import time
import pickle
import math
import array
import numpy as np
import ROOT
import constants as C
import units as U
import material as mat
import particle as prt
import bins
import fluctuations as flct
import model
import hist

import matplotlib.pyplot as plt

EE = 50 # MeV
LL = 10 # um
print(f"Model with energy: {EE} [MeV], dx: {LL} [um]")


start = time.time()
### calculate the model's parameters
dEdxModel  = "G4:Tcut" # or "BB:Tcut"
TargetMat  = mat.Si # or e.g. mat.Al
PrimaryPrt = prt.Particle(name="proton",meV=938.27208816*U.MeV2eV,mamu=1.007276466621,chrg=+1.,spin=0.5,lepn=0,magm=2.79284734463)
par        = flct.Parameters(PrimaryPrt,TargetMat,dEdxModel,"inputs/dEdx_p_si.txt")
modelpars  = par.GetModelPars(EE*U.MeV2eV,LL*U.um2cm)
print(modelpars)

### Build the model shapes
DOTIME = True
Mod = model.Model(LL*U.um2cm, EE*U.MeV2eV, modelpars, DOTIME)
# Mod.set_fft_sampling_pars(N_t_bins=10000000,frac=0.01)
Mod.set_fft_sampling_pars_rotem(N_t_bins=10000000,frac=0.01)
Mod.set_all_shapes()

end = time.time()
elapsed = end - start
print(f"Elapsed time is {elapsed} [s]")

### these are all the possible outputs (besides some other class variables)
# cnt_pdfs              = Mod.cnt_pdfs ## dict name-->TH1D
# cnt_cdfs              = Mod.cnt_cdfs ## dict name-->TH1D
# cnt_pdfs_scaled       = Mod.cnt_pdfs_scaled ## dict name-->TH1D
# cnt_cdfs_scaled       = Mod.cnt_cdfs_scaled ## dict name-->TH1D
# cnt_pdfs_scaled_arrx  = Mod.cnt_pdfs_scaled_arrx  ## np.array
# cnt_pdfs_scaled_arrsy = Mod.cnt_pdfs_scaled_arrsy ## dict name-->np.array
cnt_cdfs_scaled_arrx  = Mod.cnt_cdfs_scaled_arrx  ## np.array
cnt_cdfs_scaled_arrsy = Mod.cnt_cdfs_scaled_arrsy ## dict name-->np.array
plt.plot(cnt_cdfs_scaled_arrx, cnt_cdfs_scaled_arrsy["hModel"])
plt.title('Minimum viable example for continuous model: '+Mod.build)
plt.xscale('log')
plt.yscale('log')
plt.ylim(3e-4,2)
plt.show()

sec_cdfs_arrx  = Mod.sec_cdfs_arrx  ## np.array
sec_cdfs_arrsy = Mod.sec_cdfs_arrsy ## dict name-->np.array
plt.plot(sec_cdfs_arrx, sec_cdfs_arrsy["hBorysov_Sec"])
plt.title('Minimum viable example for secondary model: '+Mod.build)
plt.xscale('log')
plt.yscale('log')
plt.ylim(4e-3,2)
plt.show()
