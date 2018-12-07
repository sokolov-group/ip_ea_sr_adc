import sys
import numpy as np
from functools import reduce
import pyscf.ao2mo
import direct_adc.direct_adc_compute as direct_adc_compute


class DirectADC:
    def __init__(self, mf):

        print ("Initializing Direct ADC...\n")

        # General info
        self.mo = mf.mo_coeff.copy()
        self.nmo = self.mo.shape[1]
        self.nelec = mf.mol.nelectron
        self.enuc = mf.mol.energy_nuc()
        self.mo_energy = mf.mo_energy.copy()
        self.e_scf = mf.e_tot
 
        # General info (spin-orbital)
        self.nmo_so = 2 * self.nmo
        self.nocc_so = mf.mol.nelectron
        self.nvir_so = self.nmo_so - self.nocc_so
        self.mo_energy_so = np.zeros(self.nmo_so)
        self.mo_energy_so[::2] = self.mo_energy.copy()
        self.mo_energy_so[1::2] = self.mo_energy.copy()

        # Direct ADC specific variables
        self.nstates = 10
        self.step = 0.01
        self.freq_range = (-1.0, 0.0)
        self.broadening = 0.01
        self.tol = 1e-6
        self.maxiter = 200
        self.method = "adc(2)"

        # Integrals
        mo_to_so = np.zeros((self.nmo, self.nmo_so))
        mo_to_so[:,::2] = self.mo.copy()
        mo_to_so[:,1::2] = self.mo.copy()

        h1e_ao = mf.get_hcore()
        self.h1e_so = reduce(np.dot, (mo_to_so.T, h1e_ao, mo_to_so))
        self.h1e_so[::2,1::2] = 0.0
        self.h1e_so[1::2,::2] = 0.0

        self.v2e_so = pyscf.ao2mo.general(mf._eri, (mo_to_so, mo_to_so, mo_to_so, mo_to_so), compact=False)
        self.v2e_so = self.v2e_so.reshape(self.nmo_so, self.nmo_so, self.nmo_so, self.nmo_so)
        self.v2e_so = self.zero_spin_cases(self.v2e_so)
        self.v2e_so -= self.v2e_so.transpose(3,1,2,0)
        self.v2e_so = np.ascontiguousarray(self.v2e_so.transpose(3,0,2,1))

    def zero_spin_cases(self, v2e):
    
        v2e[::2,1::2] = v2e[1::2,::2] = v2e[:,:,::2,1::2] = v2e[:,:,1::2,::2] = 0.0
        
        return v2e

    def kernel(self):

        direct_adc_compute.kernel(self)

