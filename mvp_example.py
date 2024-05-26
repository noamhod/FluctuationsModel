import time
import pickle
import math
import array
import numpy as np
import ROOT
import constants as C
import units as U
import material as mat
import bins
import fluctuations as flct
import model
import hist

import matplotlib.pyplot as plt

EE = 50 # MeV
XX = 10 # um
WW = 0.01 # % of EE and XX for the slice
print(f"Model with energy: {EE} [MeV], dx: {XX} [um], window: {WW*100} [%]")


start = time.time()
### calculate the model's parameters
TargetMat = mat.Si # or e.g. mat.Al
ParticleN = "Proton"
ParticleM = C.mp
ParticleQ = +1
ParamName = ParticleN+"_on_"+TargetMat.name
dEdxModel = "G4:Tcut" # or "BB:Tcut"
par       = flct.Parameters(ParamName,ParticleM,ParticleQ,TargetMat,dEdxModel,"inputs/eloss_p_si.txt","inputs/BB.csv")
modelpars = par.GetModelPars(EE*U.MeV2eV,XX*U.um2cm)
print(modelpars)
got_params = time.time()
elapsed_params = got_params - start

### Build the model shapes
Mod = model.Model(XX*U.um2cm, EE*U.MeV2eV, modelpars)
defined_model = time.time()
elapsed_model = defined_model - got_params
Mod.set_all_shapes()
got_shapes = time.time()
elapsed_shapes = got_shapes - defined_model


print(f"Elapsed times are")
print(f"- get parameters: {elapsed_params} [s]")
print(f"- define model:   {elapsed_model} [s]")
print(f"- get shapes:     {elapsed_shapes} [s]")

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
plt.show()

sec_cdfs_arrx  = Mod.sec_cdfs_arrx  ## np.array
sec_cdfs_arrsy = Mod.sec_cdfs_arrsy ## dict name-->np.array
plt.plot(sec_cdfs_arrx, sec_cdfs_arrsy["hBorysov_Sec"])
plt.title('Minimum viable example for secondary model: '+Mod.build)
plt.show()


