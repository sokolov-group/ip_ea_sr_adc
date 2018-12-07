import sys
import numpy as np
from functools import reduce
import pyscf.ao2mo
import direct_adc.direct_adc_compute as direct_adc_compute


class DirectADC:
    def __init__(self, mf):

        print ("Initializing Direct ADC...\n")

        # General info
        self.mo_a = mf.mo_coeff.copy()
        self.mo_b = mf.mo_coeff.copy()
        self.nmo = self.mo_a.shape[1]
        self.nelec = mf.mol.nelectron
        self.nelec_a = mf.mol.nelectron // 2
        self.nelec_b = mf.mol.nelectron // 2
        self.nocc_a = self.nelec_a
        self.nocc_b = self.nelec_b
        self.nvir_a = self.nmo - self.nelec_a
        self.nvir_b = self.nmo - self.nelec_b
        self.enuc = mf.mol.energy_nuc()
        self.mo_energy_a = mf.mo_energy.copy()
        self.mo_energy_b = mf.mo_energy.copy()
        self.e_scf = mf.e_tot
 
        # Direct ADC specific variables
        self.nstates = 10
        self.step = 0.01
        self.freq_range = (-1.0, 0.0)
        self.broadening = 0.01
        self.tol = 1e-6
        self.maxiter = 200
        self.method = "adc(2)"

        # Integral transformation
        h1e_ao = mf.get_hcore()
        self.h1e_a = reduce(np.dot, (self.mo_a.T, h1e_ao, self.mo_a))
        self.h1e_b = reduce(np.dot, (self.mo_b.T, h1e_ao, self.mo_b))

        self.v2e = lambda:None

        occ_a = self.mo_a[:,:self.nocc_a].copy()
        occ_b = self.mo_b[:,:self.nocc_b].copy()
        vir_a = self.mo_a[:,self.nocc_a:].copy()
        vir_b = self.mo_a[:,self.nocc_b:].copy()

        v2e_vvvv_a = pyscf.ao2mo.general(mf._eri, (vir_a, vir_a, vir_a, vir_a), compact=False)
        v2e_vvvv_a = v2e_vvvv_a.transpose(0,2,1,3).copy()
        v2e_vvvv_a -= v2e_vvvv_a.transpose(0,1,3,2)

        v2e_vvvv_ab = pyscf.ao2mo.general(mf._eri, (vir_a, vir_a, vir_b, vir_b), compact=False)
        v2e_vvvv_ab = v2e_vvvv_ab.transpose(0,2,1,3).copy()
        v2e_vvvv_b = pyscf.ao2mo.general(mf._eri, (vir_b, vir_b, vir_b, vir_b), compact=False)
        v2e_vvvv_b = v2e_vvvv_b.transpose(0,2,1,3).copy()
        v2e_vvvv_b -= v2e_vvvv_b.transpose(0,1,3,2)

        self.v2e.vvvv = (v2e_vvvv_a, v2e_vvvv_ab, v2e_vvvv_b)
        print(v2e_vvvv_a.flags)
        print(v2e_vvvv_ab.flags)
        print(v2e_vvvv_b.flags)
        exit()


    def kernel(self):

        direct_adc_compute.kernel(self)

