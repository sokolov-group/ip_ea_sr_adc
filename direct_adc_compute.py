import sys
import numpy as np
import time
from functools import reduce

###########################################
# Computing dynamical IP-ADC #
###########################################
def kernel(direct_adc):

    if (direct_adc.method != "adc(2)" and direct_adc.method != "adc(3)" and direct_adc.method != "adc(2)-e"):
        raise Exception("Method is unknown")

    np.set_printoptions(linewidth=150, edgeitems=10,threshold=100000, suppress=True)

    print ("\nStarting spin-orbital direct ADC code..\n")
    print ("Number of electrons:               ", direct_adc.nelec)
    print ("Number of alpha electrons:         ", direct_adc.nelec_a)
    print ("Number of beta electrons:          ", direct_adc.nelec_b)
    print ("Number of alpha basis functions:         ", direct_adc.nmo_a)
    print ("Number of beta basis functions:         ", direct_adc.nmo_b)
    print ("Number of alpha occupied orbitals: ", direct_adc.nocc_a)
    print ("Number of beta occupied orbitals:  ", direct_adc.nocc_b)
    print ("Number of alpha virtual orbitals:  ", direct_adc.nvir_a)
    print ("Number of beta virtual orbitals:   ", direct_adc.nvir_b)
    print ("Nuclear repulsion energy:          ", direct_adc.enuc,"\n")
    print ("Number of states:            ", direct_adc.nstates)
    print ("Frequency step:              ", direct_adc.step)
    print ("Frequency range:             ", direct_adc.freq_range)
    print ("Broadening:                  ", direct_adc.broadening)
    print ("Tolerance:                   ", direct_adc.tol)
    print ("Maximum number of iterations:", direct_adc.maxiter)
    print ("IP:", direct_adc.IP)
    print ("EA:", direct_adc.EA, "\n")
    print ("SCF orbital energies(alpha):\n", direct_adc.mo_energy_a, "\n")
    print ("SCF orbital energies(beta):\n", direct_adc.mo_energy_b, "\n")

    t_start = time.time()

    # Compute amplitudes
    t_amp = compute_amplitudes(direct_adc)
    
    # Compute MP2 energy
    e_mp2 = compute_mp2_energy(direct_adc, t_amp)
     
    print ("MP2 correlation energy:  ", e_mp2)
    print ("MP2 total energy:        ", (direct_adc.e_scf + e_mp2), "\n")

    apply_H_ip = None
    apply_H_ea = None
    precond_ip = None
    precond_ea = None
    M_ij = None
    M_ab = None

    if direct_adc.IP == True:

        # Compute the sigma vector,preconditioner and guess vector
        apply_H_ip, precond_ip, M_ij = define_H_ip(direct_adc,t_amp)

    if direct_adc.EA == True:

        # Compute the sigma vector for EA
        apply_H_ea, precond_ea, M_ab = define_H_ea(direct_adc,t_amp)

    # Compute Green's functions directly
    #dos,orbital = calc_density_of_states(direct_adc, apply_H,precond,t_amp)
    dos_ip, dos_ea = calc_density_of_states(direct_adc, apply_H_ip, apply_H_ea, precond_ip, precond_ea,t_amp)

    if direct_adc.IP == True and direct_adc.EA == False:
        np.savetxt('density_of_states_ip.txt', dos_ip, fmt='%.8f')
    if direct_adc.EA == True and direct_adc.IP == False:
        np.savetxt('density_of_states_ea.txt', dos_ea, fmt='%.8f')

    density  = [sum(x) for x in zip(dos_ip, dos_ea)]


    if direct_adc.IP == True and direct_adc.EA == True:
        np.savetxt('total_density_of_states.txt',density, fmt='%.8f') 


    #np.savetxt('orbital_info_3s_adc3', orbital, fmt='%.8f')

    print ("Computation successfully finished")
    print ("Total time:", (time.time() - t_start, "sec"))
    
###########################################
# Computing conventional IP/EA-ADC #
###########################################
#@profile
def conventional(direct_adc):

    if (direct_adc.method != "adc(2)" and direct_adc.method != "adc(3)" and direct_adc.method != "adc(2)-e"):
        raise Exception("Method is unknown")

    np.set_printoptions(linewidth=150, edgeitems=10,threshold=100000, suppress=True)

    print ("\nStarting spin-orbital direct ADC code..\n")
    print ("Number of electrons:               ", direct_adc.nelec)
    print ("Number of alpha electrons:         ", direct_adc.nelec_a)
    print ("Number of beta electrons:          ", direct_adc.nelec_b)
    print ("Number of alpha basis functions:         ", direct_adc.nmo_a)
    print ("Number of beta basis functions:         ", direct_adc.nmo_b)
    print ("Number of alpha occupied orbitals: ", direct_adc.nocc_a)
    print ("Number of beta occupied orbitals:  ", direct_adc.nocc_b)
    print ("Number of alpha virtual orbitals:  ", direct_adc.nvir_a)
    print ("Number of beta virtual orbitals:   ", direct_adc.nvir_b)
    print ("Number of states:                  ", direct_adc.nstates)
    print ("Nuclear repulsion energy:          ", direct_adc.enuc,"\n")
    print ("SCF orbital energies(alpha):\n", direct_adc.mo_energy_a, "\n")
    print ("SCF orbital energies(beta):\n", direct_adc.mo_energy_b, "\n")

    t_start = time.time()

    # Compute amplitudes
    t_amp = compute_amplitudes(direct_adc)
    
    print ("time to calculate amplitudes:", (time.time() - t_start, "sec"))
    
    # Compute MP2 energy
    e_mp2 = compute_mp2_energy(direct_adc, t_amp)
     
    print ("MP2 correlation energy:  ", e_mp2)
    print ("MP2 total energy:        ", (direct_adc.e_scf + e_mp2), "\n")


    # Compute the sigma vector,preconditioner and guess vector

    apply_H_ip = None
    apply_H_ea = None
    precond_ip = None
    precond_ea = None

    if direct_adc.IP == True:
        apply_H_ip, precond_ip, x0_ip = setup_davidson_ip(direct_adc, t_amp)
        

    if direct_adc.EA == True:
        apply_H_ea, precond_ea, x0_ea = setup_davidson_ea(direct_adc, t_amp)


    # Compute ionization energies using Davidson

    if direct_adc.IP == True:
        E_ip, U_ip = direct_adc.davidson(apply_H_ip, x0_ip, precond_ip, nroots = direct_adc.nstates, verbose = direct_adc.verbose, max_cycle = direct_adc.max_cycle, max_space = direct_adc.max_space)

    if direct_adc.EA == True:
        E_ea, U_ea = direct_adc.davidson(apply_H_ea, x0_ea, precond_ea, nroots = direct_adc.nstates, verbose = direct_adc.verbose, max_cycle = direct_adc.max_cycle, max_space = direct_adc.max_space)

    if direct_adc.IP == True:
        print ("\n%s ionization energies (a.u.):" % (direct_adc.method))
        print (E_ip.reshape(-1, 1))
        print ("\n%s ionization energies (eV):" % (direct_adc.method))
        E_ip_ev = E_ip * 27.211606
        print (E_ip_ev.reshape(-1, 1))


    if direct_adc.EA == True:
        print ("\n%s attachment energies (a.u.):" % (direct_adc.method))
        #print (E_ea.reshape(-1, 1))
        print (E_ea.reshape(-1, 1))
        print ("\n%s attachment energies (eV):" % (direct_adc.method))
        E_ea_ev = E_ea * 27.211606
        print (E_ea_ev.reshape(-1, 1))


    # Compute transition moments and spectroscopic factors

    if direct_adc.IP == True:
        P = spec_factors_ip(direct_adc, t_amp, U_ip)
        print ("%s spectroscopic intensity:" % (direct_adc.method))
        print (P.reshape(-1,1))

    if direct_adc.EA == True:
        P = spec_factors_ea(direct_adc, t_amp, U_ea)
        print ("%s spectroscopic intensity:" % (direct_adc.method))
        print (P.reshape(-1,1))

    print ("Computation successfully finished")
    print ("Total time:", (time.time() - t_start, "sec"))

###########################################
# Computing MOM-conventional IP-ADC #
###########################################

def mom_conventional(direct_adc):

    if (direct_adc.method != "adc(2)" and direct_adc.method != "adc(3)" and direct_adc.method != "adc(2)-e"):
        raise Exception("Method is unknown")

    np.set_printoptions(linewidth=150, edgeitems=10,threshold=100000, suppress=True)

    print ("\nStarting spin-orbital direct ADC code..\n")
    print ("Number of electrons:               ", direct_adc.nelec)
    print ("Number of alpha electrons:         ", direct_adc.nelec_a)
    print ("Number of beta electrons:          ", direct_adc.nelec_b)
    print ("Number of basis functions:         ", direct_adc.nmo)
    print ("Number of alpha occupied orbitals: ", direct_adc.nocc_a)
    print ("Number of beta occupied orbitals:  ", direct_adc.nocc_b)
    print ("Number of alpha virtual orbitals:  ", direct_adc.nvir_a)
    print ("Number of beta virtual orbitals:   ", direct_adc.nvir_b)
    print ("Number of states:                  ", direct_adc.nstates)
    print ("Nuclear repulsion energy:          ", direct_adc.enuc,"\n")
    print ("SCF orbital energies(alpha):\n", direct_adc.mo_energy_a, "\n")
    print ("SCF orbital energies(beta):\n", direct_adc.mo_energy_b, "\n")

    t_start = time.time()

    # Compute amplitudes
    t_amp = compute_amplitudes(direct_adc)

    # Compute MP2 energy
    e_mp2 = compute_mp2_energy(direct_adc, t_amp)

    print ("MP2 correlation energy:  ", e_mp2)
    print ("MP2 total energy:        ", (direct_adc.e_scf + e_mp2), "\n")

    # Compute the sigma vector,preconditioner and guess vector
    apply_H, precond, x0 = setup_davidson(direct_adc, t_amp)

    # Run CVS-Davidson calculations
    E_cvs, U_cvs = direct_adc.davidson(apply_H, x0, precond, nroots = direct_adc.nstates, verbose = direct_adc.verbose, max_cycle = direct_adc.max_cycle, max_space = direct_adc.max_space)
    
    #Compute function for filtering states
    nroots = direct_adc.nstates
    U_cvs = np.array(U_cvs)
    pick_mom_states = filter_states(direct_adc, U_cvs, nroots)

    # Compute ionization energies using Davidson
    direct_adc.algorithm = "conventional"
    E, U = direct_adc.davidson(apply_H, U_cvs, precond, nroots = direct_adc.nstates, verbose = direct_adc.verbose, max_cycle = direct_adc.max_cycle, max_space = direct_adc.max_space, pick = pick_mom_states)
    index = E.argsort()
 
    print ("\n%s ionization energies (a.u.):" % (direct_adc.method))
    print (E[index].reshape(-1, 1))
    print ("\n%s ionization energies (eV):" % (direct_adc.method))
    E_ev = E * 27.2114
    print (E_ev[index].reshape(-1, 1))

    print ("Computation successfully finished")
    print ("Total time:", (time.time() - t_start, "sec"))

###########################################
# Calculate t-amplitudes  #
###########################################
#@profile
def compute_amplitudes(direct_adc):

    t2_1, t2_2, t1_2, t1_3 = (None,) * 4

    nocc_a = direct_adc.nocc_a
    nocc_b = direct_adc.nocc_b
    nvir_a = direct_adc.nvir_a
    nvir_b = direct_adc.nvir_b

    v2e_oovv_a,v2e_oovv_ab,v2e_oovv_b  = direct_adc.v2e.oovv
    v2e_vvvv_a,v2e_vvvv_ab,v2e_vvvv_b  = direct_adc.v2e.vvvv
    v2e_oooo_a,v2e_oooo_ab,v2e_oooo_b  = direct_adc.v2e.oooo
    v2e_voov_a,v2e_voov_ab,v2e_voov_b  = direct_adc.v2e.voov
    v2e_ooov_a,v2e_ooov_ab,v2e_ooov_b  = direct_adc.v2e.ooov
    v2e_vovv_a,v2e_vovv_ab,v2e_vovv_b  = direct_adc.v2e.vovv
    v2e_vvoo_a,v2e_vvoo_ab,v2e_vvoo_b  = direct_adc.v2e.vvoo
    v2e_oovo_a,v2e_oovo_ab,v2e_oovo_b  = direct_adc.v2e.oovo
    v2e_ovov_a,v2e_ovov_ab,v2e_ovov_b  = direct_adc.v2e.ovov
    v2e_vovo_a,v2e_vovo_ab,v2e_vovo_b  = direct_adc.v2e.vovo
    v2e_vvvo_a,v2e_vvvo_ab,v2e_vvvo_b  = direct_adc.v2e.vvvo
    v2e_vvov_a,v2e_vvov_ab,v2e_vvov_b  = direct_adc.v2e.vvov
    v2e_vooo_a,v2e_vooo_ab,v2e_vooo_b  = direct_adc.v2e.vooo
    v2e_ovoo_a,v2e_ovoo_ab,v2e_ovoo_b  = direct_adc.v2e.ovoo
    v2e_ovvv_a,v2e_ovvv_ab,v2e_ovvv_b  = direct_adc.v2e.ovvv
    v2e_oovo_a,v2e_oovo_ab,v2e_oovo_b  = direct_adc.v2e.oovo
    v2e_ovvo_a,v2e_ovvo_ab,v2e_ovvo_b  = direct_adc.v2e.ovvo

    e_a = direct_adc.mo_energy_a
    e_b = direct_adc.mo_energy_b

    d_ij_a = e_a[:nocc_a][:,None] + e_a[:nocc_a]
    d_ij_b = e_b[:nocc_b][:,None] + e_b[:nocc_b]
    d_ij_ab = e_a[:nocc_a][:,None] + e_b[:nocc_b]
    
    d_ab_a = e_a[nocc_a:][:,None] + e_a[nocc_a:]
    d_ab_b = e_b[nocc_b:][:,None] + e_b[nocc_b:]
    d_ab_ab = e_a[nocc_a:][:,None] + e_b[nocc_b:]
    
    D2_a = d_ij_a.reshape(-1,1) - d_ab_a.reshape(-1)
    D2_b = d_ij_b.reshape(-1,1) - d_ab_b.reshape(-1)
    D2_ab = d_ij_ab.reshape(-1,1) - d_ab_ab.reshape(-1)
    
    D2_a = D2_a.reshape((nocc_a,nocc_a,nvir_a,nvir_a))
    D2_b = D2_b.reshape((nocc_b,nocc_b,nvir_b,nvir_b))
    D2_ab = D2_ab.reshape((nocc_a,nocc_b,nvir_a,nvir_b))
        
    D1_a = e_a[:nocc_a][:None].reshape(-1,1) - e_a[nocc_a:].reshape(-1)
    D1_b = e_b[:nocc_b][:None].reshape(-1,1) - e_b[nocc_b:].reshape(-1)
    D1_a = D1_a.reshape((nocc_a,nvir_a))
    D1_b = D1_b.reshape((nocc_b,nvir_b))
    
############ Compute t2_1, t1_2 ##############################

    t2_1_a = v2e_oovv_a/D2_a
    t2_1_b = v2e_oovv_b/D2_b
    t2_1_ab = v2e_oovv_ab/D2_ab

    t2_1 = (t2_1_a , t2_1_ab, t2_1_b)    

    t1_2_a = 0.5*np.einsum('akcd,ikcd->ia',v2e_vovv_a,t2_1_a)
    t1_2_a -= 0.5*np.einsum('klic,klac->ia',v2e_ooov_a,t2_1_a)
    
    t1_2_a += np.einsum('akcd,ikcd->ia',v2e_vovv_ab,t2_1_ab)
    t1_2_a -= np.einsum('klic,klac->ia',v2e_ooov_ab,t2_1_ab)
    
    t1_2_b = 0.5*np.einsum('akcd,ikcd->ia',v2e_vovv_b,t2_1_b)
    t1_2_b -= 0.5*np.einsum('klic,klac->ia',v2e_ooov_b,t2_1_b)
    
    t1_2_b += np.einsum('kadc,kidc->ia',v2e_ovvv_ab,t2_1_ab)
    t1_2_b -= np.einsum('lkci,lkca->ia',v2e_oovo_ab,t2_1_ab)

    t1_2_a = t1_2_a/D1_a
    t1_2_b = t1_2_b/D1_b

    t1_2 = (t1_2_a , t1_2_b)
    
############ Compute t2_2 ##############################

    if (direct_adc.method == "adc(2)-e" or direct_adc.method == "adc(3)"):

        #print("Calculating additional amplitudes for adc(2)-e and adc(3)")

        temp = t2_1_a.reshape(nocc_a*nocc_a,nvir_a*nvir_a)
        temp_1 = v2e_vvvv_a[:].reshape(nvir_a*nvir_a,nvir_a*nvir_a)
        t2_2_a = 0.5*np.dot(temp,temp_1.T).reshape(nocc_a,nocc_a,nvir_a,nvir_a)
        del temp_1
        t2_2_a += 0.5*np.einsum('klij,klab->ijab',v2e_oooo_a,t2_1_a,optimize=True)
                 
        temp = np.einsum('bkjc,kica->ijab',v2e_voov_a,t2_1_a,optimize=True) 
        temp_1 = np.einsum('bkjc,ikac->ijab',v2e_voov_ab,t2_1_ab,optimize=True) 
        
        t2_2_a += temp - temp.transpose(1,0,2,3) - temp.transpose(0,1,3,2) + temp.transpose(1,0,3,2)
        t2_2_a += temp_1 - temp_1.transpose(1,0,2,3) - temp_1.transpose(0,1,3,2) + temp_1.transpose(1,0,3,2)
        
        temp = t2_1_b.reshape(nocc_b*nocc_b,nvir_b*nvir_b)
        temp_1 = v2e_vvvv_b[:].reshape(nvir_b*nvir_b,nvir_b*nvir_b)
        t2_2_b = 0.5*np.dot(temp,temp_1.T).reshape(nocc_b,nocc_b,nvir_b,nvir_b)
        del temp_1
        t2_2_b += 0.5*np.einsum('klij,klab->ijab',v2e_oooo_b,t2_1_b,optimize=True)

        temp = np.einsum('bkjc,kica->ijab',v2e_voov_b,t2_1_b,optimize=True) 
        temp_1 = np.einsum('kbcj,kica->ijab',v2e_ovvo_ab,t2_1_ab,optimize=True) 
        
        t2_2_b += temp - temp.transpose(1,0,2,3) - temp.transpose(0,1,3,2) + temp.transpose(1,0,3,2)
        t2_2_b += temp_1 - temp_1.transpose(1,0,2,3) - temp_1.transpose(0,1,3,2) + temp_1.transpose(1,0,3,2)

        temp = t2_1_ab.reshape(nocc_a*nocc_b,nvir_a*nvir_b)
        temp_1 = v2e_vvvv_ab[:].reshape(nvir_a*nvir_b,nvir_a*nvir_b)
        t2_2_ab = np.dot(temp,temp_1.T).reshape(nocc_a,nocc_b,nvir_a,nvir_b)
        del temp_1
        t2_2_ab += np.einsum('klij,klab->ijab',v2e_oooo_ab,t2_1_ab,optimize=True)
        
        t2_2_ab += np.einsum('kbcj,kica->ijab',v2e_ovvo_ab,t2_1_a,optimize=True) 
        t2_2_ab += np.einsum('bkjc,ikac->ijab',v2e_voov_b,t2_1_ab,optimize=True) 
        
        t2_2_ab -= np.einsum('kbic,kjac->ijab',v2e_ovov_ab,t2_1_ab,optimize=True) 
        
        t2_2_ab -= np.einsum('akcj,ikcb->ijab',v2e_vovo_ab,t2_1_ab,optimize=True) 

        t2_2_ab += np.einsum('akic,kjcb->ijab',v2e_voov_ab,t2_1_b,optimize=True)
        t2_2_ab += np.einsum('akic,kjcb->ijab',v2e_voov_a,t2_1_ab,optimize=True)
        
        t2_2_a = t2_2_a/D2_a
        t2_2_b = t2_2_b/D2_b
        t2_2_ab = t2_2_ab/D2_ab
        
        t2_2 = (t2_2_a , t2_2_ab, t2_2_b)    

############ Compute t1_3 ##############################

    if (direct_adc.method == "adc(3)"):
         
        #print("Calculating additional amplitudes for adc(3)")
        
        t1_3_a = np.einsum('d,ilad,ld->ia',e_a[nocc_a:],t2_1_a,t1_2_a,optimize=True)
        t1_3_a += np.einsum('d,ilad,ld->ia',e_b[nocc_b:],t2_1_ab,t1_2_b,optimize=True)
        
        t1_3_b  = np.einsum('d,ilad,ld->ia',e_b[nocc_b:],t2_1_b, t1_2_b,optimize=True)
        t1_3_b += np.einsum('d,lida,ld->ia',e_a[nocc_a:],t2_1_ab,t1_2_a,optimize=True)
        
        t1_3_a -= np.einsum('l,ilad,ld->ia',e_a[:nocc_a],t2_1_a, t1_2_a,optimize=True)
        t1_3_a -= np.einsum('l,ilad,ld->ia',e_b[:nocc_b],t2_1_ab,t1_2_b,optimize=True)
        
        t1_3_b -= np.einsum('l,ilad,ld->ia',e_b[:nocc_b],t2_1_b, t1_2_b,optimize=True)
        t1_3_b -= np.einsum('l,lida,ld->ia',e_a[:nocc_a],t2_1_ab,t1_2_a,optimize=True)

        t1_3_a += 0.5*np.einsum('a,ilad,ld->ia',e_a[nocc_a:],t2_1_a, t1_2_a,optimize=True)
        t1_3_a += 0.5*np.einsum('a,ilad,ld->ia',e_a[nocc_a:],t2_1_ab,t1_2_b,optimize=True)
        
        t1_3_b += 0.5*np.einsum('a,ilad,ld->ia',e_b[nocc_b:],t2_1_b, t1_2_b,optimize=True)
        t1_3_b += 0.5*np.einsum('a,lida,ld->ia',e_b[nocc_b:],t2_1_ab,t1_2_a,optimize=True)

        t1_3_a -= 0.5*np.einsum('i,ilad,ld->ia',e_a[:nocc_a],t2_1_a, t1_2_a,optimize=True)
        t1_3_a -= 0.5*np.einsum('i,ilad,ld->ia',e_a[:nocc_a],t2_1_ab,t1_2_b,optimize=True)
        
        t1_3_b -= 0.5*np.einsum('i,ilad,ld->ia',e_b[:nocc_b],t2_1_b, t1_2_b,optimize=True)
        t1_3_b -= 0.5*np.einsum('i,lida,ld->ia',e_b[:nocc_b],t2_1_ab,t1_2_a,optimize=True)

        t1_3_a += np.einsum('ld,adil->ia',t1_2_a,v2e_vvoo_a ,optimize=True)
        t1_3_a += np.einsum('ld,adil->ia',t1_2_b,v2e_vvoo_ab,optimize=True)
        
        t1_3_b += np.einsum('ld,adil->ia',t1_2_b,v2e_vvoo_b ,optimize=True)
        t1_3_b += np.einsum('ld,dali->ia',t1_2_a,v2e_vvoo_ab,optimize=True)

        t1_3_a += np.einsum('ld,alid->ia',t1_2_a,v2e_voov_a ,optimize=True)
        t1_3_a += np.einsum('ld,alid->ia',t1_2_b,v2e_voov_ab,optimize=True)

        t1_3_b += np.einsum('ld,alid->ia',t1_2_b,v2e_voov_b ,optimize=True)
        t1_3_b += np.einsum('ld,ladi->ia',t1_2_a,v2e_ovvo_ab,optimize=True)

        t1_3_a -= 0.5*np.einsum('lmad,lmid->ia',t2_2_a,v2e_ooov_a,optimize=True)
        t1_3_a -=     np.einsum('lmad,lmid->ia',t2_2_ab,v2e_ooov_ab,optimize=True)
        
        t1_3_b -= 0.5*np.einsum('lmad,lmid->ia',t2_2_b,v2e_ooov_b,optimize=True)
        t1_3_b -=     np.einsum('mlda,mldi->ia',t2_2_ab,v2e_oovo_ab,optimize=True)

        t1_3_a += 0.5*np.einsum('ilde,alde->ia',t2_2_a,v2e_vovv_a,optimize=True)
        t1_3_a += np.einsum('ilde,alde->ia',t2_2_ab,v2e_vovv_ab,optimize=True)
        
        t1_3_b += 0.5*np.einsum('ilde,alde->ia',t2_2_b,v2e_vovv_b,optimize=True)
        t1_3_b += np.einsum('lied,laed->ia',t2_2_ab,v2e_ovvv_ab,optimize=True)

        t1_3_a -= np.einsum('ildf,aefm,lmde->ia',t2_1_a,v2e_vvvo_a,  t2_1_a ,optimize=True)
        t1_3_a += np.einsum('ilfd,aefm,mled->ia',t2_1_ab,v2e_vvvo_a, t2_1_ab,optimize=True)
        t1_3_a -= np.einsum('ildf,aefm,lmde->ia',t2_1_a,v2e_vvvo_ab, t2_1_ab,optimize=True)
        t1_3_a += np.einsum('ilfd,aefm,lmde->ia',t2_1_ab,v2e_vvvo_ab,t2_1_b ,optimize=True)
        t1_3_a -= np.einsum('ildf,aemf,mlde->ia',t2_1_ab,v2e_vvov_ab,t2_1_ab,optimize=True)

        t1_3_b -= np.einsum('ildf,aefm,lmde->ia',t2_1_b,v2e_vvvo_b,t2_1_b,optimize=True)
        t1_3_b += np.einsum('lidf,aefm,lmde->ia',t2_1_ab,v2e_vvvo_b,t2_1_ab,optimize=True)
        t1_3_b -= np.einsum('ildf,eamf,mled->ia',t2_1_b,v2e_vvov_ab,t2_1_ab,optimize=True)
        t1_3_b += np.einsum('lidf,eamf,lmde->ia',t2_1_ab,v2e_vvov_ab,t2_1_a,optimize=True)
        t1_3_b -= np.einsum('lifd,eafm,lmed->ia',t2_1_ab,v2e_vvvo_ab,t2_1_ab,optimize=True)

        t1_3_a += 0.5*np.einsum('ilaf,defm,lmde->ia',t2_1_a,v2e_vvvo_a,t2_1_a,optimize=True)
        t1_3_a += 0.5*np.einsum('ilaf,defm,lmde->ia',t2_1_ab,v2e_vvvo_b,t2_1_b,optimize=True)
        t1_3_a += np.einsum('ilaf,edmf,mled->ia',t2_1_ab,v2e_vvov_ab,t2_1_ab,optimize=True)
        t1_3_a += np.einsum('ilaf,defm,lmde->ia',t2_1_a,v2e_vvvo_ab,t2_1_ab,optimize=True)

        t1_3_b += 0.5*np.einsum('ilaf,defm,lmde->ia',t2_1_b,v2e_vvvo_b,t2_1_b,optimize=True)
        t1_3_b += 0.5*np.einsum('lifa,defm,lmde->ia',t2_1_ab,v2e_vvvo_a,t2_1_a,optimize=True)
        t1_3_b += np.einsum('lifa,defm,lmde->ia',t2_1_ab,v2e_vvvo_ab,t2_1_ab,optimize=True)
        t1_3_b += np.einsum('ilaf,edmf,mled->ia',t2_1_b,v2e_vvov_ab,t2_1_ab,optimize=True)

        t1_3_a += 0.25*np.einsum('inde,anlm,lmde->ia',t2_1_a,v2e_vooo_a,t2_1_a,optimize=True)
        t1_3_a += np.einsum('inde,anlm,lmde->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab,optimize=True)

        t1_3_b += 0.25*np.einsum('inde,anlm,lmde->ia',t2_1_b,v2e_vooo_b,t2_1_b,optimize=True)
        t1_3_b += np.einsum('nied,naml,mled->ia',t2_1_ab,v2e_ovoo_ab,t2_1_ab,optimize=True)

        t1_3_a += 0.5*np.einsum('inad,enlm,lmde->ia',t2_1_a,v2e_vooo_a,t2_1_a,optimize=True)
        t1_3_a -= 0.5 * np.einsum('inad,neml,mlde->ia',t2_1_a,v2e_ovoo_ab,t2_1_ab,optimize=True)
        t1_3_a -= 0.5 * np.einsum('inad,nelm,lmde->ia',t2_1_a,v2e_ovoo_ab,t2_1_ab,optimize=True)
        t1_3_a -= 0.5 *np.einsum('inad,enlm,lmed->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab,optimize=True)
        t1_3_a -= 0.5*np.einsum('inad,enml,mled->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab,optimize=True)
        t1_3_a += 0.5*np.einsum('inad,enlm,lmde->ia',t2_1_ab,v2e_vooo_b,t2_1_b,optimize=True)
        
        t1_3_b += 0.5*np.einsum('inad,enlm,lmde->ia',t2_1_b,v2e_vooo_b,t2_1_b,optimize=True)
        t1_3_b -= 0.5 * np.einsum('inad,enml,mled->ia',t2_1_b,v2e_vooo_ab,t2_1_ab,optimize=True)
        t1_3_b -= 0.5 * np.einsum('inad,enlm,lmed->ia',t2_1_b,v2e_vooo_ab,t2_1_ab,optimize=True)
        t1_3_b -= 0.5 *np.einsum('nida,nelm,lmde->ia',t2_1_ab,v2e_ovoo_ab,t2_1_ab,optimize=True)
        t1_3_b -= 0.5*np.einsum('nida,neml,mlde->ia',t2_1_ab,v2e_ovoo_ab,t2_1_ab,optimize=True)
        t1_3_b += 0.5*np.einsum('nida,enlm,lmde->ia',t2_1_ab,v2e_vooo_a,t2_1_a,optimize=True)

        t1_3_a -= 0.5*np.einsum('lnde,amin,lmde->ia',t2_1_a,v2e_vooo_a,t2_1_a,optimize=True)
        t1_3_a -= np.einsum('nled,amin,mled->ia',t2_1_ab,v2e_vooo_a,t2_1_ab,optimize=True)
        t1_3_a -= 0.5*np.einsum('lnde,amin,lmde->ia',t2_1_b,v2e_vooo_ab,t2_1_b,optimize=True)
        t1_3_a -= np.einsum('lnde,amin,lmde->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab,optimize=True)

        t1_3_b -= 0.5*np.einsum('lnde,amin,lmde->ia',t2_1_b,v2e_vooo_b,t2_1_b,optimize=True)
        t1_3_b -= np.einsum('lnde,amin,lmde->ia',t2_1_ab,v2e_vooo_b,t2_1_ab,optimize=True)
        t1_3_b -= 0.5*np.einsum('lnde,mani,lmde->ia',t2_1_a,v2e_ovoo_ab,t2_1_a,optimize=True)
        t1_3_b -= np.einsum('nled,mani,mled->ia',t2_1_ab,v2e_ovoo_ab,t2_1_ab,optimize=True)

        t1_3_a += 0.5*np.einsum('lmdf,afie,lmde->ia',t2_1_a,v2e_vvov_a,t2_1_a,optimize=True)
        t1_3_a += np.einsum('mlfd,afie,mled->ia',t2_1_ab,v2e_vvov_a,t2_1_ab,optimize=True)
        t1_3_a += 0.5*np.einsum('lmdf,afie,lmde->ia',t2_1_b,v2e_vvov_ab,t2_1_b,optimize=True)
        t1_3_a += np.einsum('lmdf,afie,lmde->ia',t2_1_ab,v2e_vvov_ab,t2_1_ab,optimize=True)
        
        t1_3_b += 0.5*np.einsum('lmdf,afie,lmde->ia',t2_1_b,v2e_vvov_b,t2_1_b,optimize=True)
        t1_3_b += np.einsum('lmdf,afie,lmde->ia',t2_1_ab,v2e_vvov_b,t2_1_ab,optimize=True)
        t1_3_b += 0.5*np.einsum('lmdf,faei,lmde->ia',t2_1_a,v2e_vvvo_ab,t2_1_a,optimize=True)
        t1_3_b += np.einsum('mlfd,faei,mled->ia',t2_1_ab,v2e_vvvo_ab,t2_1_ab,optimize=True)
        
        t1_3_a -= np.einsum('lnde,emin,lmad->ia',t2_1_a,v2e_vooo_a,t2_1_a,optimize=True)
        t1_3_a += np.einsum('lnde,mein,lmad->ia',t2_1_ab,v2e_ovoo_ab,t2_1_a,optimize=True)
        t1_3_a += np.einsum('nled,emin,mlad->ia',t2_1_ab,v2e_vooo_a,t2_1_ab,optimize=True)
        t1_3_a += np.einsum('lned,emin,lmad->ia',t2_1_ab,v2e_vooo_ab,t2_1_ab,optimize=True)
        t1_3_a -= np.einsum('lnde,mein,mlad->ia',t2_1_b,v2e_ovoo_ab,t2_1_ab,optimize=True)

        t1_3_b -= np.einsum('lnde,emin,lmad->ia',t2_1_b,v2e_vooo_b,t2_1_b,optimize=True)
        t1_3_b += np.einsum('nled,emni,lmad->ia',t2_1_ab,v2e_vooo_ab,t2_1_b,optimize=True)
        t1_3_b += np.einsum('lnde,emin,lmda->ia',t2_1_ab,v2e_vooo_b,t2_1_ab,optimize=True)
        t1_3_b += np.einsum('nlde,meni,mlda->ia',t2_1_ab,v2e_ovoo_ab,t2_1_ab,optimize=True)
        t1_3_b -= np.einsum('lnde,emni,lmda->ia',t2_1_a,v2e_vooo_ab,t2_1_ab,optimize=True)

        t1_3_a -= 0.25*np.einsum('lmef,efid,lmad->ia',t2_1_a,v2e_vvov_a,t2_1_a,optimize=True)
        t1_3_a -= np.einsum('lmef,efid,lmad->ia',t2_1_ab,v2e_vvov_ab,t2_1_ab,optimize=True)
        
        t1_3_b -= 0.25*np.einsum('lmef,efid,lmad->ia',t2_1_b,v2e_vvov_b,t2_1_b,optimize=True)
        temp = t2_1_ab.reshape(nocc_a*nocc_b,-1)
        temp_1 = v2e_vvvo_ab.reshape(nvir_a*nvir_b,-1)
        temp_2 = t2_1_ab.reshape(nocc_a*nocc_b*nvir_a,-1)
        int_1 = np.dot(temp,temp_1).reshape(nocc_a*nocc_b*nvir_a,-1)
        t1_3_b -= np.dot(int_1.T,temp_2).reshape(nocc_b,nvir_b) 
        
        t1_3_a = t1_3_a/D1_a
        t1_3_b = t1_3_b/D1_b
    
        t1_3 = (t1_3_a , t1_3_b)    

    t_amp = (t2_1, t2_2, t1_2, t1_3)

    return t_amp

###########################################
# Calculate mp2 energy  #
########################################### 
##@profile
def compute_mp2_energy(direct_adc, t_amp):

    v2e_oovv_a, v2e_oovv_ab, v2e_oovv_b = direct_adc.v2e.oovv
    
    t2_1_a, t2_1_ab, t2_1_b  = t_amp[0]
    
    e_mp2 = 0.25 * np.einsum('ijab,ijab', t2_1_a, v2e_oovv_a)
    e_mp2 += np.einsum('ijab,ijab', t2_1_ab, v2e_oovv_ab)
    e_mp2 += 0.25 * np.einsum('ijab,ijab', t2_1_b, v2e_oovv_b)
  
    return e_mp2

###########################################
# Calculate density of states  #
########################################### 

def calc_density_of_states(direct_adc,apply_H_ip,apply_H_ea,precond_ip,precond_ea,t_amp):

	nmo_a = direct_adc.nmo_a
	nmo_b = direct_adc.nmo_b
	nocc_a = direct_adc.nocc_a
	freq_range = direct_adc.freq_range	
	broadening = direct_adc.broadening
	step = direct_adc.step

	freq_range = np.arange(freq_range[0],freq_range[1],step)

	k_a = np.zeros((nmo_a,nmo_a))
	k_b = np.zeros((nmo_b,nmo_b))
	gf_ip_a = np.array(k_a,dtype = complex)
	gf_ip_a.imag = k_a
	gf_ip_b = np.array(k_b,dtype = complex)
	gf_ip_b.imag = k_b
	gf_ea_a = np.array(k_a,dtype = complex)
	gf_ea_a.imag = k_a
	gf_ea_b = np.array(k_b,dtype = complex)
	gf_ea_b.imag = k_b

	gf_ip_a_trace= []
	gf_ip_b_trace= []
	gf_ea_a_trace= []
	gf_ea_b_trace= []
	gf_ip_trace = []
	gf_ea_trace = []
	gf_ip_im_trace = [] 
	gf_ea_im_trace = [] 
	orbital = []
        
	for freq in freq_range:
       
		omega = freq
		iomega = freq + broadening*1j

		for orb in range(nmo_a):
                       
                        # Calculate GF for IP
                        if direct_adc.IP == True:

		             # Calculate T and GF for alpha spin	
                        
                             T_a = calculate_T_ip(direct_adc, t_amp, orb, spin = "alpha")
                             gf_ip_a[orb,orb] = calculate_GF_ip(direct_adc,apply_H_ip,precond_ip,omega,orb,T_a)

                        # Calculate GF for EA
                        if direct_adc.EA == True:

		             # Calculate T and GF for alpha spin	
                        
                             T_a = calculate_T_ea(direct_adc, t_amp, orb, spin = "alpha")
                             gf_ea_a[orb,orb] = calculate_GF_ea(direct_adc,apply_H_ea,precond_ea,omega,orb,T_a)

		for orb in range(nmo_b):
                       
                        # Calculate GF for IP
                        if direct_adc.IP == True:

		              # Calculate T and GF for beta spin	
			
                             T_b = calculate_T_ip(direct_adc, t_amp, orb, spin = "beta")
                             gf_ip_b[orb,orb] = calculate_GF_ip(direct_adc,apply_H_ip,precond_ip,omega,orb,T_b)

                        # Calculate GF for EA
                        if direct_adc.EA == True:

		              # Calculate T and GF for beta spin	
			
                             T_b = calculate_T_ea(direct_adc, t_amp, orb, spin = "beta")
                             gf_ea_b[orb,orb] = calculate_GF_ea(direct_adc,apply_H_ea,precond_ea,omega,orb,T_b)

		gf_ip_a_trace = -(1/(np.pi))*np.trace(gf_ip_a.imag)
		gf_ip_b_trace = -(1/(np.pi))*np.trace(gf_ip_b.imag)
		gf_ea_a_trace = -(1/(np.pi))*np.trace(gf_ea_a.imag)
		gf_ea_b_trace = -(1/(np.pi))*np.trace(gf_ea_b.imag)
		gf_ip_trace = np.sum([gf_ip_a_trace,gf_ip_b_trace])
		gf_ea_trace = np.sum([gf_ea_a_trace,gf_ea_b_trace])
		gf_ip_im_trace.append(gf_ip_trace)
		gf_ea_im_trace.append(gf_ea_trace)

	return gf_ip_im_trace,gf_ea_im_trace         

##############################################
# Calculate Transition moments matrix for IP #
############################################## 
def calculate_T_ip(direct_adc, t_amp, orb, spin=None):

    t_start = time.time()

    method = direct_adc.method

    t2_1, t2_2, t1_2, t1_3 = t_amp

    t2_1_a, t2_1_ab, t2_1_b = t2_1
    t1_2_a, t1_2_b = t1_2

    nocc_a = direct_adc.nocc_a
    nocc_b = direct_adc.nocc_b
    nvir_a = direct_adc.nvir_a
    nvir_b = direct_adc.nvir_b
    
    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)

    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a* (nocc_a - 1) * nvir_a // 2 
    n_doubles_bab = nvir_b * nocc_a* nocc_b  
    n_doubles_aba = nvir_a * nocc_b* nocc_a
    n_doubles_bbb = nocc_b* (nocc_b - 1) * nvir_b // 2
    
    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)
    
    v2e_oovv_a , v2e_oovv_ab, v2e_oovv_b = direct_adc.v2e.oovv
    v2e_vvvo_a , v2e_vvvo_ab, v2e_vvvo_b = direct_adc.v2e.vvvo
    v2e_ovoo_a , v2e_ovoo_ab, v2e_ovoo_b = direct_adc.v2e.ovoo
    v2e_voov_a , v2e_voov_ab, v2e_voov_b = direct_adc.v2e.voov
    v2e_ovov_a , v2e_ovov_ab, v2e_ovov_b = direct_adc.v2e.ovov
    v2e_vovv_a , v2e_vovv_ab, v2e_vovv_b = direct_adc.v2e.vovv
    v2e_ooov_a , v2e_ooov_ab, v2e_ooov_b = direct_adc.v2e.ooov
    
    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b 
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb    

    T = np.zeros((dim))
    
######## spin = alpha  ############################################

    if spin=="alpha":
   
######## ADC(2) 1h part  ############################################

        if orb < nocc_a:
            T[s_a:f_a]  = idn_occ_a[orb, :]
            T[s_a:f_a] += 0.25*np.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:], t2_1_a, optimize = True)
            T[s_a:f_a] -= 0.25*np.einsum('kdc,ikdc->i',t2_1_ab[orb,:,:,:], t2_1_ab, optimize = True)
            T[s_a:f_a] -= 0.25*np.einsum('kcd,ikcd->i',t2_1_ab[orb,:,:,:], t2_1_ab, optimize = True)
        else :    
            T[s_a:f_a] += t1_2_a[:,(orb-nocc_a)]
        
######## ADC(2) 2h-1p  part  ############################################

            t2_1_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:].copy()
            t2_1_t_a = t2_1_t.transpose(2,1,0).copy()           
            
            t2_1_t_ab = t2_1_ab.transpose(2,3,1,0).copy()           

            T[s_aaa:f_aaa] = t2_1_t_a[(orb-nocc_a),:,:].reshape(-1)
            T[s_bab:f_bab] = t2_1_t_ab[(orb-nocc_a),:,:,:].reshape(-1)
            
######## ADC(3) 2h-1p  part  ############################################

        if(method=='adc(2)-e'or method=='adc(3)'):
        
            t2_2_a, t2_2_ab, t2_2_b = t2_2
   
            if orb >= nocc_a:
                t2_2_t = t2_2_a[ij_ind_a[0],ij_ind_a[1],:,:].copy()
                t2_2_t_a = t2_2_t.transpose(2,1,0).copy()           
                
                t2_2_t_ab = t2_2_ab.transpose(2,3,1,0).copy()           
   
                T[s_aaa:f_aaa] += t2_2_t_a[(orb-nocc_a),:,:].reshape(-1)
                T[s_bab:f_bab] += t2_2_t_ab[(orb-nocc_a),:,:,:].reshape(-1)

######## ADC(3) 1h part  ############################################

        if(method=='adc(3)'):
    	
            t1_3_a, t1_3_b = t1_3         

            if orb < nocc_a:
                T[s_a:f_a] += 0.25*np.einsum('kdc,ikdc->i',t2_1_a[:,orb,:,:], t2_2_a, optimize = True)
                T[s_a:f_a] -= 0.25*np.einsum('kdc,ikdc->i',t2_1_ab[orb,:,:,:], t2_2_ab, optimize = True)
                T[s_a:f_a] -= 0.25*np.einsum('kcd,ikcd->i',t2_1_ab[orb,:,:,:], t2_2_ab, optimize = True)

                T[s_a:f_a] += 0.25*np.einsum('ikdc,kdc->i',t2_1_a, t2_2_a[:,orb,:,:],optimize = True) 
                T[s_a:f_a] -= 0.25*np.einsum('ikcd,kcd->i',t2_1_ab, t2_2_ab[orb,:,:,:],optimize = True) 
                T[s_a:f_a] -= 0.25*np.einsum('ikdc,kdc->i',t2_1_ab, t2_2_ab[orb,:,:,:],optimize = True) 
            else: 
                T[s_a:f_a] += 0.5*np.einsum('ikc,kc->i',t2_1_a[:,:,(orb-nocc_a),:], t1_2_a,optimize = True)
                T[s_a:f_a] += 0.5*np.einsum('ikc,kc->i',t2_1_ab[:,:,(orb-nocc_a),:], t1_2_b,optimize = True)
                T[s_a:f_a] += t1_3_a[:,(orb-nocc_a)]

######## spin = beta  ############################################

    if spin=="beta":
   
######## ADC(2) 1h part  ############################################

        if orb < nocc_b:
            T[s_b:f_b] = idn_occ_b[orb, :]
            T[s_b:f_b]+= 0.25*np.einsum('kdc,ikdc->i',t2_1_b[:,orb,:,:], t2_1_b, optimize = True)
            T[s_b:f_b]-= 0.25*np.einsum('kdc,kidc->i',t2_1_ab[:,orb,:,:], t2_1_ab, optimize = True)
            T[s_b:f_b]-= 0.25*np.einsum('kcd,kicd->i',t2_1_ab[:,orb,:,:], t2_1_ab, optimize = True)
        else :    
            T[s_b:f_b] += t1_2_b[:,(orb-nocc_b)]
        
######## ADC(2) 2h-1p part  ############################################

            t2_1_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:].copy()
            t2_1_t_b = t2_1_t.transpose(2,1,0).copy()           

            t2_1_t_ab = t2_1_ab.transpose(2,3,0,1).copy()           

            T[s_bbb:f_bbb] = t2_1_t_b[(orb-nocc_b),:,:].reshape(-1)
            T[s_aba:f_aba] = t2_1_t_ab[:,(orb-nocc_b),:,:].reshape(-1)

######## ADC(3) 2h-1p part  ############################################

        if(method=='adc(2)-e'or method=='adc(3)'):

            t2_2_a, t2_2_ab, t2_2_b = t2_2

            if orb >= nocc_b:
                t2_2_t = t2_2_b[ij_ind_b[0],ij_ind_b[1],:,:].copy()
                t2_2_t_b = t2_2_t.transpose(2,1,0).copy()           

                t2_2_t_ab = t2_2_ab.transpose(2,3,0,1).copy()           

                T[s_bbb:f_bbb] += t2_2_t_b[(orb-nocc_b),:,:].reshape(-1)
                T[s_aba:f_aba] += t2_2_t_ab[:,(orb-nocc_b),:,:].reshape(-1)
                   
######## ADC(2) 1h part  ############################################

        if(method=='adc(3)'):
    	
            t1_3_a, t1_3_b = t1_3         
            t2_2_a, t2_2_ab, t2_2_b = t2_2

            if orb < nocc_b:
                T[s_b:f_b] += 0.25*np.einsum('kdc,ikdc->i',t2_1_b[:,orb,:,:], t2_2_b, optimize = True)
                T[s_b:f_b] -= 0.25*np.einsum('kdc,kidc->i',t2_1_ab[:,orb,:,:], t2_2_ab, optimize = True)
                T[s_b:f_b] -= 0.25*np.einsum('kcd,kicd->i',t2_1_ab[:,orb,:,:], t2_2_ab, optimize = True)

                T[s_b:f_b] += 0.25*np.einsum('ikdc,kdc->i',t2_1_b, t2_2_b[:,orb,:,:],optimize = True) 
                T[s_b:f_b] -= 0.25*np.einsum('kicd,kcd->i',t2_1_ab, t2_2_ab[:,orb,:,:],optimize = True) 
                T[s_b:f_b] -= 0.25*np.einsum('kidc,kdc->i',t2_1_ab, t2_2_ab[:,orb,:,:],optimize = True) 
            else: 
                T[s_b:f_b] += 0.5*np.einsum('ikc,kc->i',t2_1_b[:,:,(orb-nocc_b),:], t1_2_b,optimize = True)
                T[s_b:f_b] += 0.5*np.einsum('kic,kc->i',t2_1_ab[:,:,:,(orb-nocc_b)], t1_2_a,optimize = True)
                T[s_b:f_b] += t1_3_b[:,(orb-nocc_b)]

    return T

###############################################
# Calculate Transition moments matrix for EA #
############################################### 
#@profile
def calculate_T_ea(direct_adc, t_amp, orb, spin=None):

    method = direct_adc.method

    t2_1, t2_2, t1_2, t1_3 = t_amp

    t2_1_a, t2_1_ab, t2_1_b = t2_1
    t1_2_a, t1_2_b = t1_2

    nocc_a = direct_adc.nocc_a
    nocc_b = direct_adc.nocc_b
    nvir_a = direct_adc.nvir_a
    nvir_b = direct_adc.nvir_b
    
    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    n_singles_a = nvir_a
    n_singles_b = nvir_b
    n_doubles_aaa = nvir_a* (nvir_a - 1) * nocc_a // 2 
    n_doubles_bab = nocc_b * nvir_a* nvir_b  
    n_doubles_aba = nocc_a * nvir_b* nvir_a
    n_doubles_bbb = nvir_b* (nvir_b - 1) * nocc_b // 2
    
    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)
    
    v2e_oovv_a , v2e_oovv_ab, v2e_oovv_b = direct_adc.v2e.oovv
    v2e_vvvo_a , v2e_vvvo_ab, v2e_vvvo_b = direct_adc.v2e.vvvo
    v2e_ovoo_a , v2e_ovoo_ab, v2e_ovoo_b = direct_adc.v2e.ovoo
    v2e_voov_a , v2e_voov_ab, v2e_voov_b = direct_adc.v2e.voov
    v2e_ovov_a , v2e_ovov_ab, v2e_ovov_b = direct_adc.v2e.ovov
    v2e_vovv_a , v2e_vovv_ab, v2e_vovv_b = direct_adc.v2e.vovv
    v2e_ooov_a , v2e_ooov_ab, v2e_ooov_b = direct_adc.v2e.ooov
    
    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b 
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb    

    T = np.zeros((dim))
    
######## spin = alpha  ############################################

    if spin=="alpha":
   
######## ADC(2) part  ############################################

        if orb < nocc_a:

            T[s_a:f_a] = -t1_2_a[orb,:]

            t2_1_t = t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]].copy()
            t2_1_ab_t = -t2_1_ab.transpose(1,0,2,3).copy()
                 
            T[s_aaa:f_aaa] += t2_1_t[:,orb,:].reshape(-1)
            T[s_bab:f_bab] += t2_1_ab_t[:,orb,:,:].reshape(-1)
 
        else :

            T[s_a:f_a] += idn_vir_a[(orb-nocc_a), :]
            T[s_a:f_a] -= 0.25*np.einsum('klc,klac->a',t2_1_a[:,:,(orb-nocc_a),:], t2_1_a, optimize = True)
            T[s_a:f_a] -= 0.25*np.einsum('klc,klac->a',t2_1_ab[:,:,(orb-nocc_a),:], t2_1_ab, optimize = True)
            T[s_a:f_a] -= 0.25*np.einsum('lkc,lkac->a',t2_1_ab[:,:,(orb-nocc_a),:], t2_1_ab, optimize = True)

######## ADC(3) 2p-1h  part  ############################################

        if(method=='adc(2)-e'or method=='adc(3)'):
        
            t2_2_a, t2_2_ab, t2_2_b = t2_2
   
            if orb < nocc_a:

                t2_2_t = t2_2_a[:,:,ab_ind_a[0],ab_ind_a[1]].copy()
                t2_2_ab_t = -t2_2_ab.transpose(1,0,2,3).copy()
                     
                T[s_aaa:f_aaa] += t2_2_t[:,orb,:].reshape(-1)
                T[s_bab:f_bab] += t2_2_ab_t[:,orb,:,:].reshape(-1)

######### ADC(3) 1p part  ############################################

        if(method=='adc(3)'):
    	
            t1_3_a, t1_3_b = t1_3         

            if orb < nocc_a:

                T[s_a:f_a] += 0.5*np.einsum('kac,ck->a',t2_1_a[:,orb,:,:], t1_2_a.T,optimize = True)
                T[s_a:f_a] -= 0.5*np.einsum('kac,ck->a',t2_1_ab[orb,:,:,:], t1_2_b.T,optimize = True)

                T[s_a:f_a] -= t1_3_a[orb,:]

            else:
               
                T[s_a:f_a] -= 0.25*np.einsum('klc,klac->a',t2_1_a[:,:,(orb-nocc_a),:], t2_2_a, optimize = True)
                T[s_a:f_a] -= 0.25*np.einsum('klc,klac->a',t2_1_ab[:,:,(orb-nocc_a),:], t2_2_ab, optimize = True)
                T[s_a:f_a] -= 0.25*np.einsum('lkc,lkac->a',t2_1_ab[:,:,(orb-nocc_a),:], t2_2_ab, optimize = True)

                T[s_a:f_a] -= 0.25*np.einsum('klac,klc->a',t2_1_a, t2_2_a[:,:,(orb-nocc_a),:],optimize = True) 
                T[s_a:f_a] -= 0.25*np.einsum('klac,klc->a',t2_1_ab, t2_2_ab[:,:,(orb-nocc_a),:],optimize = True) 
                T[s_a:f_a] -= 0.25*np.einsum('lkac,lkc->a',t2_1_ab, t2_2_ab[:,:,(orb-nocc_a),:],optimize = True) 

######### spin = beta  ############################################
#
    if spin=="beta":
   
######## ADC(2) part  ############################################


        if orb < nocc_b:

            T[s_b:f_b] = -t1_2_b[orb,:]

            t2_1_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]].copy()
            t2_1_ab_t = -t2_1_ab.transpose(0,1,3,2).copy()
                 
            T[s_bbb:f_bbb] += t2_1_t[:,orb,:].reshape(-1)
            T[s_aba:f_aba] += t2_1_ab_t[:,orb,:,:].reshape(-1)

        else :

            T[s_b:f_b] += idn_vir_b[(orb-nocc_b), :]
            T[s_b:f_b] -= 0.25*np.einsum('klc,klac->a',t2_1_b[:,:,(orb-nocc_b),:], t2_1_b, optimize = True)
            T[s_b:f_b] -= 0.25*np.einsum('lkc,lkca->a',t2_1_ab[:,:,:,(orb-nocc_b)], t2_1_ab, optimize = True)
            T[s_b:f_b] -= 0.25*np.einsum('lkc,lkca->a',t2_1_ab[:,:,:,(orb-nocc_b)], t2_1_ab, optimize = True)

######### ADC(3) 2p-1h part  ############################################

        if(method=='adc(2)-e'or method=='adc(3)'):

            t2_2_a, t2_2_ab, t2_2_b = t2_2

            if orb < nocc_b:
                   
                t2_2_t = t2_2_b[:,:,ab_ind_b[0],ab_ind_b[1]].copy()
                t2_2_ab_t = -t2_2_ab.transpose(0,1,3,2).copy()
                     
                T[s_bbb:f_bbb] += t2_2_t[:,orb,:].reshape(-1)
                T[s_aba:f_aba] += t2_2_ab_t[:,orb,:,:].reshape(-1)

######### ADC(2) 1p part  ############################################

        if(method=='adc(3)'):
    	
            t1_3_a, t1_3_b = t1_3         

            if orb < nocc_b:

                T[s_b:f_b] += 0.5*np.einsum('kac,ck->a',t2_1_b[:,orb,:,:], t1_2_b.T,optimize = True)
                T[s_b:f_b] -= 0.5*np.einsum('kca,ck->a',t2_1_ab[:,orb,:,:], t1_2_a.T,optimize = True)

                T[s_b:f_b] -= t1_3_b[orb,:]

            else:
               
                T[s_b:f_b] -= 0.25*np.einsum('klc,klac->a',t2_1_b[:,:,(orb-nocc_b),:], t2_2_b, optimize = True)
                T[s_b:f_b] -= 0.25*np.einsum('lkc,lkca->a',t2_1_ab[:,:,:,(orb-nocc_b)], t2_2_ab, optimize = True)
                T[s_b:f_b] -= 0.25*np.einsum('lkc,lkca->a',t2_1_ab[:,:,:,(orb-nocc_b)], t2_2_ab, optimize = True)

                T[s_b:f_b] -= 0.25*np.einsum('klac,klc->a',t2_1_b, t2_2_b[:,:,(orb-nocc_b),:],optimize = True) 
                T[s_b:f_b] -= 0.25*np.einsum('lkca,lkc->a',t2_1_ab, t2_2_ab[:,:,:,(orb-nocc_b)],optimize = True) 
                T[s_b:f_b] -= 0.25*np.einsum('klca,klc->a',t2_1_ab, t2_2_ab[:,:,:,(orb-nocc_b)],optimize = True) 
        
    return T

###########################################
# Calculate Green's Function matrix  #
########################################### 

def calculate_GF_ip(direct_adc,apply_H,precond,omega,orb,T):

    method = direct_adc.method

    nocc_a = direct_adc.nocc_a
    nocc_b = direct_adc.nocc_b
    nvir_a = direct_adc.nvir_a
    nvir_b = direct_adc.nvir_b
    
    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b 
    n_doubles_aba = nvir_a * nocc_b * nocc_a 
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b 
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb    

    broadening = direct_adc.broadening
    nocc_a = direct_adc.nocc_a
    nocc_b = direct_adc.nocc_b
    nvir_a = direct_adc.nvir_a
    nvir_b = direct_adc.nvir_b

    iomega = omega + broadening*1j
    
    imag_r = -(np.real(T))/broadening

    sigma = apply_H(imag_r)

    real_r =  (-omega*imag_r  - np.real(sigma))/broadening

    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b 
    n_doubles_aba = nvir_a * nocc_b * nocc_a 
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    z = np.zeros((dim))
    new_r = np.array(z,dtype = complex)
    new_r.imag = z

    new_r.real = real_r.copy()
    new_r.imag = imag_r.copy()
    
    new_r = solve_conjugate_gradients(direct_adc,apply_H,precond,T,new_r,omega,orb)
    
    gf = np.dot(T,new_r)

    return gf
    
def calculate_GF_ea(direct_adc,apply_H,precond,omega,orb,T):

    method = direct_adc.method

    nocc_a = direct_adc.nocc_a
    nocc_b = direct_adc.nocc_b
    nvir_a = direct_adc.nvir_a
    nvir_b = direct_adc.nvir_b
    
    n_singles_a = nvir_a
    n_singles_b = nvir_b
    n_doubles_aaa = nvir_a * (nvir_a - 1) * nocc_a // 2
    n_doubles_bab = nocc_b * nvir_a * nvir_b 
    n_doubles_aba = nocc_a * nvir_b * nvir_a 
    n_doubles_bbb = nvir_b * (nvir_b - 1) * nocc_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b 
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb    

    broadening = direct_adc.broadening

    iomega = omega + broadening*1j
    
    imag_r = -(np.real(T))/broadening

    sigma = apply_H(imag_r)

    real_r =  (-omega*imag_r  - np.real(sigma))/broadening

    z = np.zeros((dim))
    new_r = np.array(z,dtype = complex)
    new_r.imag = z

    new_r.real = real_r.copy()
    new_r.imag = imag_r.copy()
    
    new_r = solve_conjugate_gradients(direct_adc,apply_H,precond,T,new_r,omega,orb)
    
    gf = np.dot(T,new_r)

    return gf

###########################################
# Precompute M_ij block   #
########################################### 
##@profile
def get_Mij(direct_adc,t_amp):

    t_start = time.time()

    method = direct_adc.method

    t2_1, t2_2, t1_2, t1_3 = t_amp

    t2_1_a, t2_1_ab, t2_1_b = t2_1
    t1_2_a, t1_2_b = t1_2

    nocc_a = direct_adc.nocc_a
    nocc_b = direct_adc.nocc_b
    nvir_a = direct_adc.nvir_a
    nvir_b = direct_adc.nvir_b

    e_occ_a = direct_adc.mo_energy_a[:nocc_a]
    e_occ_b = direct_adc.mo_energy_b[:nocc_b]
    e_vir_a = direct_adc.mo_energy_a[nocc_a:]
    e_vir_b = direct_adc.mo_energy_b[nocc_b:]

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    v2e_oovv_a,v2e_oovv_ab,v2e_oovv_b = direct_adc.v2e.oovv
    v2e_vvoo_a,v2e_vvoo_ab,v2e_vvoo_b = direct_adc.v2e.vvoo
    v2e_ooov_a,v2e_ooov_ab,v2e_ooov_b = direct_adc.v2e.ooov
    v2e_ovoo_a,v2e_ovoo_ab,v2e_ovoo_b = direct_adc.v2e.ovoo
    v2e_ovov_a,v2e_ovov_ab,v2e_ovov_b = direct_adc.v2e.ovov
    v2e_vovo_a,v2e_vovo_ab,v2e_vovo_b = direct_adc.v2e.vovo
    v2e_oooo_a,v2e_oooo_ab,v2e_oooo_b = direct_adc.v2e.oooo
    v2e_ovvo_a,v2e_ovvo_ab,v2e_ovvo_b = direct_adc.v2e.ovvo
    v2e_vvvv_a,v2e_vvvv_ab,v2e_vvvv_b = direct_adc.v2e.vvvv
    v2e_voov_a,v2e_voov_ab,v2e_voov_b = direct_adc.v2e.voov
    v2e_oovo_a,v2e_oovo_ab,v2e_oovo_b = direct_adc.v2e.oovo
    v2e_vooo_a,v2e_vooo_ab,v2e_vooo_b = direct_adc.v2e.vooo

    # i-j block
    # Zeroth-order terms

    M_ij_a = np.einsum('ij,j->ij', idn_occ_a ,e_occ_a)
    M_ij_b = np.einsum('ij,j->ij', idn_occ_b ,e_occ_b)

    # Second-order terms

    M_ij_a +=  np.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_a, t2_1_a)
    M_ij_a +=  np.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_ab, t2_1_ab)
    M_ij_a +=  np.einsum('d,iled,jled->ij',e_vir_b,t2_1_ab, t2_1_ab)
    
    M_ij_b +=  np.einsum('d,ilde,jlde->ij',e_vir_b,t2_1_b, t2_1_b)
    M_ij_b +=  np.einsum('d,lide,ljde->ij',e_vir_a,t2_1_ab, t2_1_ab)
    M_ij_b +=  np.einsum('d,lied,ljed->ij',e_vir_b,t2_1_ab, t2_1_ab)

    M_ij_a -= 0.5 *  np.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_a, t2_1_a)
    M_ij_a -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab)
    M_ij_a -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_1_ab)

    M_ij_b -= 0.5 *  np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_b, t2_1_b)
    M_ij_b -= 0.5*np.einsum('l,lide,ljde->ij',e_occ_a,t2_1_ab, t2_1_ab)
    M_ij_b -= 0.5*np.einsum('l,lied,ljed->ij',e_occ_a,t2_1_ab, t2_1_ab)

    M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_a, t2_1_a)
    M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)
    M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)

    M_ij_b -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_b,t2_1_b, t2_1_b)
    M_ij_b -= 0.25 *  np.einsum('i,lied,ljed->ij',e_occ_b,t2_1_ab, t2_1_ab)
    M_ij_b -= 0.25 *  np.einsum('i,lide,ljde->ij',e_occ_b,t2_1_ab, t2_1_ab)

    M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_a, t2_1_a)
    M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)
    M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_1_ab)

    M_ij_b -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_b,t2_1_b, t2_1_b)
    M_ij_b -= 0.25 *  np.einsum('j,lied,ljed->ij',e_occ_b,t2_1_ab, t2_1_ab)
    M_ij_b -= 0.25 *  np.einsum('j,lide,ljde->ij',e_occ_b,t2_1_ab, t2_1_ab)

    M_ij_a += 0.5 *  np.einsum('ilde,jlde->ij',t2_1_a, v2e_oovv_a)
    M_ij_a += np.einsum('ilde,jlde->ij',t2_1_ab, v2e_oovv_ab)

    M_ij_b += 0.5 *  np.einsum('ilde,jlde->ij',t2_1_b, v2e_oovv_b)
    M_ij_b += np.einsum('lied,ljed->ij',t2_1_ab, v2e_oovv_ab)

    M_ij_a += 0.5 *  np.einsum('jlde,deil->ij',t2_1_a, v2e_vvoo_a)
    M_ij_a += np.einsum('jlde,deil->ij',t2_1_ab, v2e_vvoo_ab)

    M_ij_b += 0.5 *  np.einsum('jlde,deil->ij',t2_1_b, v2e_vvoo_b)
    M_ij_b += np.einsum('ljed,edli->ij',t2_1_ab, v2e_vvoo_ab)

    # Third-order terms
    
    if (method == "adc(3)"):

        t2_2_a, t2_2_ab, t2_2_b = t2_2
        M_ij_a += np.einsum('ld,jlid->ij',t1_2_a, v2e_ooov_a)
        M_ij_a += np.einsum('ld,jlid->ij',t1_2_b, v2e_ooov_ab)

        M_ij_b += np.einsum('ld,jlid->ij',t1_2_b, v2e_ooov_b)
        M_ij_b += np.einsum('ld,ljdi->ij',t1_2_a, v2e_oovo_ab)
        
        M_ij_a += np.einsum('ld,jdil->ij',t1_2_a, v2e_ovoo_a)
        M_ij_a += np.einsum('ld,jdil->ij',t1_2_b, v2e_ovoo_ab)

        M_ij_b += np.einsum('ld,jdil->ij',t1_2_b, v2e_ovoo_b)
        M_ij_b += np.einsum('ld,djli->ij',t1_2_a, v2e_vooo_ab)

        M_ij_a += 0.5* np.einsum('ilde,jlde->ij',t2_2_a, v2e_oovv_a)
        M_ij_a += np.einsum('ilde,jlde->ij',t2_2_ab, v2e_oovv_ab)

        M_ij_b += 0.5* np.einsum('ilde,jlde->ij',t2_2_b, v2e_oovv_b)
        M_ij_b += np.einsum('lied,ljed->ij',t2_2_ab, v2e_oovv_ab)

        M_ij_a += 0.5* np.einsum('jlde,deil->ij',t2_2_a, v2e_vvoo_a)
        M_ij_a += np.einsum('jlde,deil->ij',t2_2_ab, v2e_vvoo_ab)

        M_ij_b += 0.5* np.einsum('jlde,deil->ij',t2_2_b, v2e_vvoo_b)
        M_ij_b += np.einsum('ljed,edli->ij',t2_2_ab, v2e_vvoo_ab)

        M_ij_a +=  np.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a +=  np.einsum('d,ilde,jlde->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a +=  np.einsum('d,iled,jled->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)
        
        M_ij_b +=  np.einsum('d,ilde,jlde->ij',e_vir_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b +=  np.einsum('d,lide,ljde->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b +=  np.einsum('d,lied,ljed->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a +=  np.einsum('d,jlde,ilde->ij',e_vir_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a +=  np.einsum('d,jlde,ilde->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a +=  np.einsum('d,jled,iled->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)
        
        M_ij_b +=  np.einsum('d,jlde,ilde->ij',e_vir_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b +=  np.einsum('d,ljde,lide->ij',e_vir_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b +=  np.einsum('d,ljed,lied->ij',e_vir_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.5 *  np.einsum('l,ilde,jlde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.5*np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
    
        M_ij_b -= 0.5 *  np.einsum('l,ilde,jlde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.5*np.einsum('l,lied,ljed->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.5*np.einsum('l,lied,ljed->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.5 *  np.einsum('l,jlde,ilde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.5*np.einsum('l,jlde,ilde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.5*np.einsum('l,jlde,ilde->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_b -= 0.5 *  np.einsum('l,jlde,ilde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.5*np.einsum('l,ljed,lied->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.5*np.einsum('l,ljed,lied->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
    
        M_ij_b -= 0.25 *  np.einsum('i,ilde,jlde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.25 *  np.einsum('i,lied,ljed->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.25 *  np.einsum('i,lied,ljed->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.25 *  np.einsum('i,jlde,ilde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.25 *  np.einsum('i,jlde,ilde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.25 *  np.einsum('i,jlde,ilde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
    
        M_ij_b -= 0.25 *  np.einsum('i,jlde,ilde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.25 *  np.einsum('i,ljed,lied->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.25 *  np.einsum('i,ljed,lied->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.25 *  np.einsum('j,jlde,ilde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.25 *  np.einsum('j,jlde,ilde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.25 *  np.einsum('j,jlde,ilde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
    
        M_ij_b -= 0.25 *  np.einsum('j,jlde,ilde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.25 *  np.einsum('j,ljed,lied->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.25 *  np.einsum('j,ljed,lied->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)

        M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_a, t2_2_a,optimize=True)
        M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_a -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_a,t2_1_ab, t2_2_ab,optimize=True)
    
        M_ij_b -= 0.25 *  np.einsum('j,ilde,jlde->ij',e_occ_b,t2_1_b, t2_2_b,optimize=True)
        M_ij_b -= 0.25 *  np.einsum('j,lied,ljed->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
        M_ij_b -= 0.25 *  np.einsum('j,lied,ljed->ij',e_occ_b,t2_1_ab, t2_2_ab,optimize=True)
    
        M_ij_a -= np.einsum('lmde,jldf,fmie->ij',t2_1_a, t2_1_a, v2e_voov_a ,optimize = True)
        M_ij_a += np.einsum('mled,jlfd,fmie->ij',t2_1_ab, t2_1_ab, v2e_voov_a ,optimize = True)
        M_ij_a -= np.einsum('lmde,jldf,fmie->ij',t2_1_ab, t2_1_a, v2e_voov_ab,optimize = True)
        M_ij_a -= np.einsum('mlde,jldf,mfie->ij',t2_1_ab, t2_1_ab, v2e_ovov_ab ,optimize = True)
        M_ij_a += np.einsum('lmde,jlfd,fmie->ij',t2_1_b, t2_1_ab, v2e_voov_ab ,optimize = True)

        M_ij_b -= np.einsum('lmde,jldf,fmie->ij',t2_1_b, t2_1_b, v2e_voov_b ,optimize = True)
        M_ij_b += np.einsum('lmde,ljdf,fmie->ij',t2_1_ab, t2_1_ab, v2e_voov_b ,optimize = True)
        M_ij_b -= np.einsum('mled,jldf,mfei->ij',t2_1_ab, t2_1_b, v2e_ovvo_ab,optimize = True)
        M_ij_b -= np.einsum('lmed,ljfd,fmei->ij',t2_1_ab, t2_1_ab, v2e_vovo_ab ,optimize = True)
        M_ij_b += np.einsum('lmde,ljdf,mfei->ij',t2_1_a, t2_1_ab, v2e_ovvo_ab ,optimize = True)

        M_ij_a -= np.einsum('lmde,ildf,fmje->ij',t2_1_a, t2_1_a, v2e_voov_a ,optimize = True)
        M_ij_a += np.einsum('mled,ilfd,fmje->ij',t2_1_ab, t2_1_ab, v2e_voov_a ,optimize = True)
        M_ij_a -= np.einsum('lmde,ildf,fmje->ij',t2_1_ab, t2_1_a, v2e_voov_ab,optimize = True)
        M_ij_a -= np.einsum('mlde,ildf,mfje->ij',t2_1_ab, t2_1_ab, v2e_ovov_ab ,optimize = True)
        M_ij_a += np.einsum('lmde,ilfd,fmje->ij',t2_1_b, t2_1_ab, v2e_voov_ab ,optimize = True)
        
        M_ij_b -= np.einsum('lmde,ildf,fmje->ij',t2_1_b, t2_1_b, v2e_voov_b ,optimize = True)
        M_ij_b += np.einsum('lmde,lidf,fmje->ij',t2_1_ab, t2_1_ab, v2e_voov_b ,optimize = True)
        M_ij_b -= np.einsum('mled,ildf,mfej->ij',t2_1_ab, t2_1_b, v2e_ovvo_ab,optimize = True)
        M_ij_b -= np.einsum('lmed,lifd,fmej->ij',t2_1_ab, t2_1_ab, v2e_vovo_ab ,optimize = True)
        M_ij_b += np.einsum('lmde,lidf,mfej->ij',t2_1_a, t2_1_ab, v2e_ovvo_ab ,optimize = True)

        M_ij_a += 0.25*np.einsum('lmde,jnde,lmin->ij',t2_1_a, t2_1_a,v2e_oooo_a, optimize = True)
        M_ij_a += np.einsum('lmde,jnde,lmin->ij',t2_1_ab ,t2_1_ab,v2e_oooo_ab, optimize = True)

        M_ij_b += 0.25*np.einsum('lmde,jnde,lmin->ij',t2_1_b, t2_1_b,v2e_oooo_b, optimize = True)
        M_ij_b += np.einsum('mled,njed,mlni->ij',t2_1_ab ,t2_1_ab,v2e_oooo_ab, optimize = True)

        M_ij_a += 0.25*np.einsum('ilde,jlgf,gfde->ij',t2_1_a, t2_1_a,v2e_vvvv_a, optimize = True)
        M_ij_a +=np.einsum('ilde,jlgf,gfde->ij',t2_1_ab, t2_1_ab,v2e_vvvv_ab, optimize = True)

        M_ij_b += 0.25*np.einsum('ilde,jlgf,gfde->ij',t2_1_b, t2_1_b,v2e_vvvv_b, optimize = True)
        M_ij_b +=np.einsum('lied,ljfg,fged->ij',t2_1_ab, t2_1_ab,v2e_vvvv_ab, optimize = True)

        M_ij_a += 0.25*np.einsum('inde,lmde,jnlm->ij',t2_1_a, t2_1_a,v2e_oooo_a, optimize = True)
        M_ij_a +=np.einsum('inde,lmde,jnlm->ij',t2_1_ab, t2_1_ab,v2e_oooo_ab, optimize = True)

        M_ij_b += 0.25*np.einsum('inde,lmde,jnlm->ij',t2_1_b, t2_1_b,v2e_oooo_b, optimize = True)
        M_ij_b +=np.einsum('nied,mled,njml->ij',t2_1_ab, t2_1_ab,v2e_oooo_ab, optimize = True)

        M_ij_a += 0.5*np.einsum('lmdf,lmde,jeif->ij',t2_1_a, t2_1_a, v2e_ovov_a , optimize = True)
        M_ij_a +=np.einsum('mlfd,mled,jeif->ij',t2_1_ab, t2_1_ab, v2e_ovov_a , optimize = True)
        M_ij_a +=np.einsum('lmdf,lmde,jeif->ij',t2_1_ab, t2_1_ab, v2e_ovov_ab , optimize = True)
        M_ij_a +=0.5*np.einsum('lmdf,lmde,jeif->ij',t2_1_b, t2_1_b, v2e_ovov_ab , optimize = True)

        M_ij_b += 0.5*np.einsum('lmdf,lmde,jeif->ij',t2_1_b, t2_1_b, v2e_ovov_b , optimize = True)
        M_ij_b +=np.einsum('lmdf,lmde,jeif->ij',t2_1_ab, t2_1_ab, v2e_ovov_b , optimize = True)
        M_ij_b +=np.einsum('lmfd,lmed,ejfi->ij',t2_1_ab, t2_1_ab, v2e_vovo_ab , optimize = True)
        M_ij_b +=0.5*np.einsum('lmdf,lmde,ejfi->ij',t2_1_a, t2_1_a, v2e_vovo_ab , optimize = True)

        M_ij_a -= np.einsum('ilde,jmdf,flem->ij',t2_1_a, t2_1_a, v2e_vovo_a, optimize = True)
        M_ij_a += np.einsum('ilde,jmdf,lfem->ij',t2_1_a, t2_1_ab, v2e_ovvo_ab, optimize = True)
        M_ij_a += np.einsum('ilde,jmdf,flme->ij',t2_1_ab, t2_1_a, v2e_voov_ab, optimize = True)
        M_ij_a -= np.einsum('ilde,jmdf,flem->ij',t2_1_ab, t2_1_ab, v2e_vovo_b, optimize = True)
        M_ij_a -= np.einsum('iled,jmfd,flem->ij',t2_1_ab, t2_1_ab, v2e_vovo_ab, optimize = True)

        M_ij_b -= np.einsum('ilde,jmdf,flem->ij',t2_1_b, t2_1_b, v2e_vovo_b, optimize = True)
        M_ij_b += np.einsum('ilde,mjfd,flme->ij',t2_1_b, t2_1_ab, v2e_voov_ab, optimize = True)
        M_ij_b += np.einsum('lied,jmdf,lfem->ij',t2_1_ab, t2_1_b, v2e_ovvo_ab, optimize = True)
        M_ij_b -= np.einsum('lied,mjfd,flem->ij',t2_1_ab, t2_1_ab, v2e_vovo_a, optimize = True)
        M_ij_b -= np.einsum('lide,mjdf,lfme->ij',t2_1_ab, t2_1_ab, v2e_ovov_ab, optimize = True)

        M_ij_a -= 0.5*np.einsum('lnde,lmde,jnim->ij',t2_1_a, t2_1_a, v2e_oooo_a, optimize = True)
        M_ij_a -= np.einsum('nled,mled,jnim->ij',t2_1_ab, t2_1_ab, v2e_oooo_a, optimize = True)
        M_ij_a -= np.einsum('lnde,lmde,jnim->ij',t2_1_ab, t2_1_ab, v2e_oooo_ab, optimize = True)
        M_ij_a -= 0.5 * np.einsum('lnde,lmde,jnim->ij',t2_1_b, t2_1_b, v2e_oooo_ab, optimize = True)

        M_ij_b -= 0.5*np.einsum('lnde,lmde,jnim->ij',t2_1_b, t2_1_b, v2e_oooo_b, optimize = True)
        M_ij_b -= np.einsum('lnde,lmde,jnim->ij',t2_1_ab, t2_1_ab, v2e_oooo_b, optimize = True)
        M_ij_b -= np.einsum('nled,mled,njmi->ij',t2_1_ab, t2_1_ab, v2e_oooo_ab, optimize = True)
        M_ij_b -= 0.5 * np.einsum('lnde,lmde,njmi->ij',t2_1_a, t2_1_a, v2e_oooo_ab, optimize = True)
    
    M_ij = (M_ij_a, M_ij_b)

    return M_ij

###########################################
# Compute sigma vector #
########################################### 
def define_H_ip(direct_adc,t_amp):

    method = direct_adc.method

    t2_1, t2_2, t1_2, t1_3 = t_amp

    t2_1_a, t2_1_ab, t2_1_b = t2_1
    t1_2_a, t1_2_b = t1_2

    nocc_a = direct_adc.nocc_a
    nocc_b = direct_adc.nocc_b
    nvir_a = direct_adc.nvir_a
    nvir_b = direct_adc.nvir_b
    
    ij_ind_a = np.tril_indices(nocc_a, k=-1)
    ij_ind_b = np.tril_indices(nocc_b, k=-1)

    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b 
    n_doubles_aba = nvir_a * nocc_b * nocc_a 
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    e_occ_a = direct_adc.mo_energy_a[:nocc_a]
    e_occ_b = direct_adc.mo_energy_b[:nocc_b]
    e_vir_a = direct_adc.mo_energy_a[nocc_a:]
    e_vir_b = direct_adc.mo_energy_b[nocc_b:]

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    v2e_oovv_a,v2e_oovv_ab,v2e_oovv_b = direct_adc.v2e.oovv
    v2e_vooo_a,v2e_vooo_ab,v2e_vooo_b = direct_adc.v2e.vooo
    v2e_oovo_a,v2e_oovo_ab,v2e_oovo_b = direct_adc.v2e.oovo
    v2e_vvoo_a,v2e_vvoo_ab,v2e_vvoo_b = direct_adc.v2e.vvoo
    v2e_ooov_a,v2e_ooov_ab,v2e_ooov_b = direct_adc.v2e.ooov
    v2e_ovoo_a,v2e_ovoo_ab,v2e_ovoo_b = direct_adc.v2e.ovoo
    v2e_vovv_a,v2e_vovv_ab,v2e_vovv_b = direct_adc.v2e.vovv
    v2e_vovo_a,v2e_vovo_ab,v2e_vovo_b = direct_adc.v2e.vovo
    v2e_oooo_a,v2e_oooo_ab,v2e_oooo_b = direct_adc.v2e.oooo
    v2e_vvvo_a,v2e_vvvo_ab,v2e_vvvo_b = direct_adc.v2e.vvvo
    v2e_ovov_a,v2e_ovov_ab,v2e_ovov_b = direct_adc.v2e.ovov
    v2e_ovvv_a,v2e_ovvv_ab,v2e_ovvv_b = direct_adc.v2e.ovvv
    v2e_vvov_a,v2e_vvov_ab,v2e_vvov_b = direct_adc.v2e.vvov
    v2e_ovvo_a,v2e_ovvo_ab,v2e_ovvo_b = direct_adc.v2e.ovvo
    v2e_voov_a,v2e_voov_ab,v2e_voov_b = direct_adc.v2e.voov

    v2e_vooo_1_a = v2e_vooo_a[:,:,ij_ind_a[0],ij_ind_a[1]].transpose(1,0,2).reshape(nocc_a,-1)
    v2e_vooo_1_b = v2e_vooo_b[:,:,ij_ind_b[0],ij_ind_b[1]].transpose(1,0,2).reshape(nocc_b,-1)

    v2e_vooo_1_ab_a = -v2e_ovoo_ab.transpose(0,1,3,2).reshape(nocc_a, -1)
    v2e_vooo_1_ab_b = -v2e_vooo_ab.transpose(1,0,2,3).reshape(nocc_b, -1)

    v2e_oovo_1_a = v2e_oovo_a[ij_ind_a[0],ij_ind_a[1],:,:].transpose(1,0,2)
    v2e_oovo_1_b = v2e_oovo_b[ij_ind_b[0],ij_ind_b[1],:,:].transpose(1,0,2)
    v2e_oovo_1_ab = -v2e_ovoo_ab.transpose(1,3,2,0)
    v2e_oovo_2_ab = -v2e_vooo_ab.transpose(0,2,3,1)

    d_ij_a = e_occ_a[:,None] + e_occ_a
    d_a_a = e_vir_a[:,None]
    D_n_a = -d_a_a + d_ij_a.reshape(-1)
    D_n_a = D_n_a.reshape((nvir_a,nocc_a,nocc_a))
    D_aij_a = D_n_a.copy()[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

    d_ij_b = e_occ_b[:,None] + e_occ_b
    d_a_b = e_vir_b[:,None]
    D_n_b = -d_a_b + d_ij_b.reshape(-1)
    D_n_b = D_n_b.reshape((nvir_b,nocc_b,nocc_b))
    D_aij_b = D_n_b.copy()[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)
    
    d_ij_ab = e_occ_b[:,None] + e_occ_a
    d_a_b = e_vir_b[:,None]
    D_n_bab = -d_a_b + d_ij_ab.reshape(-1)
    D_aij_bab = D_n_bab.reshape(-1)

    d_ij_ab = e_occ_a[:,None] + e_occ_b
    d_a_a = e_vir_a[:,None]
    D_n_aba = -d_a_a + d_ij_ab.reshape(-1)
    D_aij_aba = D_n_aba.reshape(-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b 
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb    
    
    M_ij_a, M_ij_b = get_Mij(direct_adc,t_amp)
    
    precond = np.zeros(dim)
    
    # Compute precond in h1-h1 block
    M_ij_a_diag = np.diagonal(M_ij_a)
    M_ij_b_diag = np.diagonal(M_ij_b)

    precond[s_a:f_a] = M_ij_a_diag.copy()
    precond[s_b:f_b] = M_ij_b_diag.copy()

    # Compute precond in 2p1h-2p1h block
  
    precond[s_aaa:f_aaa] = D_aij_a.copy()
    precond[s_bab:f_bab] = D_aij_bab.copy()
    precond[s_aba:f_aba] = D_aij_aba.copy()
    precond[s_bbb:f_bbb] = D_aij_b.copy()

    # Compute preconditioner for CVS

    if direct_adc.algorithm == "cvs" or direct_adc.algorithm == "mom_conventional":

        shift = -100000.0
        ncore = direct_adc.n_core

        precond[(s_a+ncore):f_a] += shift
        precond[(s_b+ncore):f_b] += shift

        temp = np.zeros((nvir_a, nocc_a, nocc_a))
        temp[:,ij_ind_a[0],ij_ind_a[1]] = precond[s_aaa:f_aaa].reshape(nvir_a,-1).copy()
        temp[:,ij_ind_a[1],ij_ind_a[0]] = -precond[s_aaa:f_aaa].reshape(nvir_a,-1).copy()

        temp[:,ncore:,ncore:] += shift
        temp[:,:ncore,:ncore] += shift

        precond[s_aaa:f_aaa] = temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1).copy()

        temp = precond[s_bab:f_bab].copy()
        temp = temp.reshape((nvir_b, nocc_a, nocc_b))
        temp[:,ncore:,ncore:] += shift
        temp[:,:ncore,:ncore] += shift

        precond[s_bab:f_bab] = temp.reshape(-1).copy()

        temp = precond[s_aba:f_aba].copy()
        temp = temp.reshape((nvir_a, nocc_b, nocc_a))
        temp[:,ncore:,ncore:] += shift
        temp[:,:ncore,:ncore] += shift

        precond[s_aba:f_aba] = temp.reshape(-1).copy()

        temp = np.zeros((nvir_b, nocc_b, nocc_b))
        temp[:,ij_ind_b[0],ij_ind_b[1]] = precond[s_bbb:f_bbb].reshape(nvir_b,-1).copy()
        temp[:,ij_ind_b[1],ij_ind_b[0]] = -precond[s_bbb:f_bbb].reshape(nvir_b,-1).copy()

        temp[:,ncore:,ncore:] += shift
        temp[:,:ncore,:ncore] += shift

        precond[s_bbb:f_bbb] = temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1).copy()

    #Calculate sigma vector        
    ##@profile
    def sigma_(r):

        if direct_adc.algorithm == "cvs" or direct_adc.algorithm == "mom_conventional":
            r = cvs_projector(direct_adc, r)

        s = None
        if direct_adc.algorithm == "dynamical":
            z = np.zeros((dim))
            s = np.array(z,dtype = complex)
            s.imag = z
        else:
            s = np.zeros((dim))

        r_a = r[s_a:f_a]
        r_b = r[s_b:f_b]
        r_aaa = r[s_aaa:f_aaa]
        r_bab = r[s_bab:f_bab]
        r_aba = r[s_aba:f_aba]
        r_bbb = r[s_bbb:f_bbb]

        #r_bab = r_bab.reshape(nvir_b,nocc_a,nocc_b)

############ ADC(2) ij block ############################
        
        s[s_a:f_a] = np.einsum('ij,j->i',M_ij_a,r_a) 
        s[s_b:f_b] = np.einsum('ij,j->i',M_ij_b,r_b) 
            
############ ADC(2) i - kja block #########################
       
        s[s_a:f_a] += np.einsum('ip,p->i', v2e_vooo_1_a, r_aaa, optimize = True)
        s[s_a:f_a] -= np.einsum('ip,p->i', v2e_vooo_1_ab_a, r_bab, optimize = True)

        s[s_b:f_b] += np.einsum('ip,p->i', v2e_vooo_1_b, r_bbb, optimize = True)
        s[s_b:f_b] -= np.einsum('ip,p->i', v2e_vooo_1_ab_b, r_aba, optimize = True)

################ ADC(2) ajk - i block ############################

        s[s_aaa:f_aaa] += np.einsum('api,i->ap', v2e_oovo_1_a, r_a, optimize = True).reshape(-1)
        s[s_bab:f_bab] -= np.einsum('ajki,i->ajk', v2e_oovo_1_ab, r_a, optimize = True).reshape(-1)
        s[s_aba:f_aba] -= np.einsum('ajki,i->ajk', v2e_oovo_2_ab, r_b, optimize = True).reshape(-1)
        s[s_bbb:f_bbb] += np.einsum('api,i->ap', v2e_oovo_1_b, r_b, optimize = True).reshape(-1)

################ ADC(2) ajk - bil block ############################

        s[s_aaa:f_aaa] += D_aij_a * r_aaa
        s[s_bab:f_bab] += D_aij_bab * r_bab.reshape(-1)
        s[s_aba:f_aba] += D_aij_aba * r_aba.reshape(-1)
        s[s_bbb:f_bbb] += D_aij_b * r_bbb

############### ADC(3) ajk - bil block ############################

        if (method == "adc(2)-e" or method == "adc(3)"):
       	
               t2_2_a, t2_2_ab, t2_2_b = t2_2

              #print("Calculating additional terms for adc(2)-e")	

               r_aaa = r_aaa.reshape(nvir_a,-1)
               r_bab = r_bab.reshape(nvir_b,nocc_b,nocc_a)
               r_aba = r_aba.reshape(nvir_a,nocc_a,nocc_b)
               r_bbb = r_bbb.reshape(nvir_b,-1)

               r_aaa_u = None
               if direct_adc.algorithm == "dynamical":
                   r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a),dtype=complex)
               else:
                   r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
               r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()    
               r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()   

               r_bbb_u = None
               if direct_adc.algorithm == "dynamical":
                   r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b),dtype=complex)
               else:
                   r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))

               r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()    
               r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()   

               temp = 0.5*np.einsum('jkli,ail->ajk',v2e_oooo_a,r_aaa_u ,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)

               temp = 0.5*np.einsum('jkli,ail->ajk',v2e_oooo_b,r_bbb_u,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)

               s[s_bab:f_bab] -= 0.5*np.einsum('kjil,ali->ajk',v2e_oooo_ab,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= 0.5*np.einsum('kjli,ail->ajk',v2e_oooo_ab,r_bab,optimize = True).reshape(-1)

               s[s_aba:f_aba] -= 0.5*np.einsum('jkli,ali->ajk',v2e_oooo_ab,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*np.einsum('jkil,ail->ajk',v2e_oooo_ab,r_aba,optimize = True).reshape(-1)

               temp = 0.5*np.einsum('bkal,bjl->ajk',v2e_vovo_a,r_aaa_u,optimize = True)
               temp += 0.5* np.einsum('kbal,blj->ajk',v2e_ovvo_ab,r_bab,optimize = True)
  
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)             

               s[s_bab:f_bab] += 0.5*np.einsum('kbla,bjl->ajk',v2e_ovov_ab,r_bab,optimize = True).reshape(-1)

               temp_1 = 0.5*np.einsum('bkal,bjl->ajk',v2e_vovo_b,r_bbb_u,optimize = True)
               temp_1 += 0.5*np.einsum('bkla,blj->ajk',v2e_voov_ab,r_aba,optimize = True)

               s[s_bbb:f_bbb] += temp_1[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)    

               s[s_aba:f_aba] += 0.5*np.einsum('bkal,bjl->ajk',v2e_vovo_ab,r_aba,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('bjal,bkl->ajk',v2e_vovo_a,r_aaa_u,optimize = True)
               temp -= 0.5*np.einsum('jbal,blk->ajk',v2e_ovvo_ab,r_bab,optimize = True)

               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)             

               s[s_bab:f_bab] +=  0.5*np.einsum('bjla,bkl->ajk',v2e_voov_ab,r_aaa_u,optimize = True).reshape(-1)
               s[s_bab:f_bab] +=  0.5*np.einsum('bjal,blk->ajk',v2e_vovo_b,r_bab,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('bjal,bkl->ajk',v2e_vovo_b,r_bbb_u,optimize = True)
               temp -= 0.5*np.einsum('bjla,blk->ajk',v2e_voov_ab,r_aba,optimize = True)

               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)             

               s[s_aba:f_aba] += 0.5*np.einsum('bjal,blk->ajk',v2e_vovo_a,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] += 0.5*np.einsum('jbal,bkl->ajk',v2e_ovvo_ab,r_bbb_u,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('bkai,bij->ajk',v2e_vovo_a,r_aaa_u,optimize = True)
               temp += 0.5*np.einsum('kbai,bij->ajk',v2e_ovvo_ab,r_bab,optimize = True)

               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)             

               s[s_bab:f_bab] += 0.5*np.einsum('kbia,bji->ajk',v2e_ovov_ab,r_bab,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('bkai,bij->ajk',v2e_vovo_b,r_bbb_u,optimize = True)
               temp += 0.5*np.einsum('bkia,bij->ajk',v2e_voov_ab,r_aba,optimize = True)

               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)             

               s[s_aba:f_aba] += 0.5*np.einsum('bkai,bji->ajk',v2e_vovo_ab,r_aba,optimize = True).reshape(-1)

               temp = 0.5*np.einsum('bjai,bik->ajk',v2e_vovo_a,r_aaa_u,optimize = True)
               temp -= 0.5*np.einsum('jbai,bik->ajk',v2e_ovvo_ab,r_bab,optimize = True)

               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1]].reshape(-1)             

               s[s_bab:f_bab] += 0.5*np.einsum('bjai,bik->ajk',v2e_vovo_b,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= 0.5*np.einsum('bjia,bik->ajk',v2e_voov_ab,r_aaa_u,optimize = True).reshape(-1)

               s[s_aba:f_aba] += 0.5*np.einsum('bjai,bik->ajk',v2e_vovo_a,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*np.einsum('jbai,bik->ajk',v2e_ovvo_ab,r_bbb_u,optimize = True).reshape(-1)

               temp = 0.5*np.einsum('bjai,bik->ajk',v2e_vovo_b,r_bbb_u,optimize = True)
               temp -= 0.5*np.einsum('bjia,bik->ajk',v2e_voov_ab,r_aba,optimize = True)

               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1]].reshape(-1)   

        if (method == "adc(3)"):
     
           #print("Calculating additional terms for adc(3)")	
             
################ ADC(3) i - kja block ############################

               #t2_1_a_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:]
               #temp = np.einsum('pbc,bcai->pai',t2_1_a_t,v2e_vvvo_a)
               #r_aaa = r_aaa.reshape(nvir_a,-1)
               #s[s_a:f_a] += 0.5*np.einsum('pai,ap->i',temp, r_aaa, optimize=True)

               r_aaa = r_aaa.reshape(nvir_a,-1)
               t2_1_a_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:].copy()
               temp = np.einsum('pbc,ap->abc',t2_1_a_t,r_aaa, optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('abc,bcai->i',temp, v2e_vvvo_a, optimize=True)

               temp_1 = np.einsum('kjcb,ajk->abc',t2_1_ab,r_bab, optimize=True)
               s[s_a:f_a] += np.einsum('abc,cbia->i',temp_1, v2e_vvov_ab, optimize=True)

               #t2_1_b_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:]
               #temp = np.einsum('pbc,bcai->pai',t2_1_b_t,v2e_vvvo_b)
               #r_bbb = r_bbb.reshape(nvir_b,-1)
               #s[s_b:f_b] += 0.5*np.einsum('pai,ap->i',temp, r_bbb, optimize=True)

               r_bbb = r_bbb.reshape(nvir_b,-1)
               t2_1_b_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:].copy()
               temp = np.einsum('pbc,ap->abc',t2_1_b_t,r_bbb, optimize=True)
               s[s_b:f_b] += 0.5*np.einsum('abc,bcai->i',temp, v2e_vvvo_b, optimize=True)

               temp_1 = np.einsum('jkbc,ajk->abc',t2_1_ab,r_aba, optimize=True)
               s[s_b:f_b] += np.einsum('abc,bcai->i',temp_1, v2e_vvvo_ab, optimize=True)

               if direct_adc.algorithm == "dynamical":
                   r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a),dtype=complex)
               else:
                   r_aaa_u = np.zeros((nvir_a,nocc_a,nocc_a))
               r_aaa_u[:,ij_ind_a[0],ij_ind_a[1]]= r_aaa.copy()    
               r_aaa_u[:,ij_ind_a[1],ij_ind_a[0]]= -r_aaa.copy()    

               if direct_adc.algorithm == "dynamical":
                   r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b),dtype=complex)
               else:
                   r_bbb_u = np.zeros((nvir_b,nocc_b,nocc_b))
               r_bbb_u[:,ij_ind_b[0],ij_ind_b[1]]= r_bbb.copy()    
               r_bbb_u[:,ij_ind_b[1],ij_ind_b[0]]= -r_bbb.copy()    

               r_bab = r_bab.reshape(nvir_b,nocc_b,nocc_a)
               r_aba = r_aba.reshape(nvir_a,nocc_a,nocc_b)

               if direct_adc.algorithm == "dynamical":
                   temp = np.zeros_like(r_bab,dtype=complex)
               else:
                   temp = np.zeros_like(r_bab)
               temp = np.einsum('jlab,ajk->blk',t2_1_a,r_aaa_u,optimize=True)
               temp += np.einsum('ljba,ajk->blk',t2_1_ab,r_bab,optimize=True)

               if direct_adc.algorithm == "dynamical":
                   temp_1 = np.zeros_like(r_bab,dtype=complex)
               else:
                   temp_1 = np.zeros_like(r_bab)
               temp_1 = np.einsum('jlab,ajk->blk',t2_1_ab,r_aaa_u,optimize=True)
               temp_1 += np.einsum('jlab,ajk->blk',t2_1_b,r_bab,optimize=True)
               
               temp_2 = np.einsum('jlba,akj->blk',t2_1_ab,r_bab, optimize=True)
               
               s[s_a:f_a] += 0.5*np.einsum('blk,ilkb->i',temp,v2e_ooov_a,optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('blk,ilkb->i',temp_1,v2e_ooov_ab,optimize=True)
               s[s_a:f_a] -= 0.5*np.einsum('blk,ilbk->i',temp_2,v2e_oovo_ab,optimize=True)

               if direct_adc.algorithm == "dynamical":
                   temp = np.zeros_like(r_aba,dtype=complex)
               else:
                   temp = np.zeros_like(r_aba)
               temp = np.einsum('jlab,ajk->blk',t2_1_b,r_bbb_u,optimize=True)
               temp += np.einsum('jlab,ajk->blk',t2_1_ab,r_aba,optimize=True)

               if direct_adc.algorithm == "dynamical":
                   temp_1 = np.zeros_like(r_aba,dtype=complex)
               else:
                   temp_1 = np.zeros_like(r_aba)
               temp_1 = np.einsum('ljba,ajk->blk',t2_1_ab,r_bbb_u,optimize=True)
               temp_1 += np.einsum('jlab,ajk->blk',t2_1_a,r_aba,optimize=True)
               
               temp_2 = np.einsum('ljab,akj->blk',t2_1_ab,r_aba,optimize=True)
               
               s[s_b:f_b] += 0.5*np.einsum('blk,ilkb->i',temp,v2e_ooov_b,optimize=True)
               s[s_b:f_b] += 0.5*np.einsum('blk,libk->i',temp_1,v2e_oovo_ab,optimize=True)
               s[s_b:f_b] -= 0.5*np.einsum('blk,likb->i',temp_2,v2e_ooov_ab,optimize=True)

               if direct_adc.algorithm == "dynamical":
                   temp = np.zeros_like(r_bab,dtype=complex)
               else:
                   temp = np.zeros_like(r_bab)
               temp = -np.einsum('klab,akj->blj',t2_1_a,r_aaa_u,optimize=True)
               temp -= np.einsum('lkba,akj->blj',t2_1_ab,r_bab,optimize=True)
               
               if direct_adc.algorithm == "dynamical":
                   temp_1 = np.zeros_like(r_bab,dtype=complex)
               else:
                   temp_1 = np.zeros_like(r_bab)
               temp_1 = -np.einsum('klab,akj->blj',t2_1_ab,r_aaa_u,optimize=True)
               temp_1 -= np.einsum('klab,akj->blj',t2_1_b,r_bab,optimize=True)

               temp_2 = -np.einsum('klba,ajk->blj',t2_1_ab,r_bab,optimize=True)

               s[s_a:f_a] -= 0.5*np.einsum('blj,iljb->i',temp,v2e_ooov_a,optimize=True)
               s[s_a:f_a] -= 0.5*np.einsum('blj,iljb->i',temp_1,v2e_ooov_ab,optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('blj,ilbj->i',temp_2,v2e_oovo_ab,optimize=True)

               if direct_adc.algorithm == "dynamical":
                   temp = np.zeros_like(r_aba,dtype=complex)
               else:
                   temp = np.zeros_like(r_aba)
               temp = -np.einsum('klab,akj->blj',t2_1_b,r_bbb_u,optimize=True)
               temp -= np.einsum('klab,akj->blj',t2_1_ab,r_aba,optimize=True)
               
               if direct_adc.algorithm == "dynamical":
                   temp_1 = np.zeros_like(r_bab,dtype=complex)
               else:
                   temp_1 = np.zeros_like(r_bab)
               temp_1 = -np.einsum('lkba,akj->blj',t2_1_ab,r_bbb_u,optimize=True)
               temp_1 -= np.einsum('klab,akj->blj',t2_1_a,r_aba,optimize=True)

               temp_2 = -np.einsum('lkab,ajk->blj',t2_1_ab,r_aba,optimize=True)

               s[s_b:f_b] -= 0.5*np.einsum('blj,iljb->i',temp,v2e_ooov_b,optimize=True)
               s[s_b:f_b] -= 0.5*np.einsum('blj,libj->i',temp_1,v2e_oovo_ab,optimize=True)
               s[s_b:f_b] += 0.5*np.einsum('blj,lijb->i',temp_2,v2e_ooov_ab,optimize=True)

################ ADC(3) ajk - i block ############################
                
               #t2_1_a_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:]
               #temp = 0.5*np.einsum('pbc,bcai->api',t2_1_a_t,v2e_vvvo_a)
               #s[s_aaa:f_aaa] += np.einsum('api,i->ap',temp, r_a, optimize=True).reshape(-1)
    
               t2_1_a_t = t2_1_a[ij_ind_a[0],ij_ind_a[1],:,:].copy()
               temp = np.einsum('i,bcai->bca',r_a,v2e_vvvo_a,optimize=True)
               s[s_aaa:f_aaa] += 0.5*np.einsum('bca,pbc->ap',temp,t2_1_a_t,optimize=True).reshape(-1)
                 
               #temp_1 = np.einsum('kjcb,cbia->iajk',t2_1_ab,v2e_vvov_ab)
               #temp_1 = temp_1.reshape(nocc_a,-1)
               #s[s_bab:f_bab] += np.einsum('ip,i->p',temp_1, r_a, optimize=True).reshape(-1)

               temp_1 = np.einsum('i,cbia->cba',r_a,v2e_vvov_ab,optimize=True)
               s[s_bab:f_bab] += np.einsum('cba,kjcb->ajk',temp_1, t2_1_ab, optimize=True).reshape(-1)

               #t2_1_b_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:]
               #temp = 0.5*np.einsum('pbc,bcai->api',t2_1_b_t,v2e_vvvo_b)
               #s[s_bbb:f_bbb] += np.einsum('api,i->ap',temp, r_b, optimize=True).reshape(-1)

               t2_1_b_t = t2_1_b[ij_ind_b[0],ij_ind_b[1],:,:].copy()
               temp = np.einsum('i,bcai->bca',r_b,v2e_vvvo_b,optimize=True)
               s[s_bbb:f_bbb] += 0.5*np.einsum('bca,pbc->ap',temp,t2_1_b_t,optimize=True).reshape(-1)

               #temp_1 = np.einsum('jkbc,bcai->iajk',t2_1_ab,v2e_vvvo_ab)
               #temp_1 = temp_1.reshape(nocc_b,-1)
               #s[s_aba:f_aba] += np.einsum('ip,i->p',temp_1, r_b, optimize=True).reshape(-1)

               temp_1 = np.einsum('i,bcai->bca',r_b,v2e_vvvo_ab,optimize=True)
               s[s_aba:f_aba] += np.einsum('bca,jkbc->ajk',temp_1, t2_1_ab, optimize=True).reshape(-1)

               temp_1 = np.einsum('i,kbil->kbl',r_a, v2e_ovoo_a)
               temp_2 = np.einsum('i,kbil->kbl',r_a, v2e_ovoo_ab)
              
               temp  = np.einsum('kbl,jlab->ajk',temp_1,t2_1_a,optimize=True)
               temp += np.einsum('kbl,jlab->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] += temp[:,ij_ind_a[0],ij_ind_a[1] ].reshape(-1)
               
               temp_1  = np.einsum('i,kbil->kbl',r_a,v2e_ovoo_a)
               temp_2  = np.einsum('i,kbil->kbl',r_a,v2e_ovoo_ab)

               temp  = np.einsum('kbl,ljba->ajk',temp_1,t2_1_ab,optimize=True)
               temp += np.einsum('kbl,jlab->ajk',temp_2,t2_1_b,optimize=True)
               s[s_bab:f_bab] += temp.reshape(-1)

               temp_1 = np.einsum('i,kbil->kbl',r_b, v2e_ovoo_b)
               temp_2 = np.einsum('i,bkli->kbl',r_b, v2e_vooo_ab)
              
               temp  = np.einsum('kbl,jlab->ajk',temp_1,t2_1_b,optimize=True)
               temp += np.einsum('kbl,ljba->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] += temp[:,ij_ind_b[0],ij_ind_b[1] ].reshape(-1)
              
               temp_1  = np.einsum('i,kbil->kbl',r_b,v2e_ovoo_b)
               temp_2  = np.einsum('i,bkli->kbl',r_b,v2e_vooo_ab)

               temp  = np.einsum('kbl,jlab->ajk',temp_1,t2_1_ab,optimize=True)
               temp += np.einsum('kbl,jlab->ajk',temp_2,t2_1_a,optimize=True)
               s[s_aba:f_aba] += temp.reshape(-1)

               temp_1 = np.einsum('i,jbil->jbl',r_a, v2e_ovoo_a)
               temp_2 = np.einsum('i,jbil->jbl',r_a, v2e_ovoo_ab)
              
               temp  = np.einsum('jbl,klab->ajk',temp_1,t2_1_a,optimize=True)
               temp += np.einsum('jbl,klab->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] -= temp[:,ij_ind_a[0],ij_ind_a[1] ].reshape(-1)
              
               temp  = -np.einsum('i,bjil->jbl',r_a,v2e_vooo_ab,optimize=True)
               temp_1 = -np.einsum('jbl,klba->ajk',temp,t2_1_ab,optimize=True)
               s[s_bab:f_bab] -= temp_1.reshape(-1)

               temp_1 = np.einsum('i,jbil->jbl',r_b, v2e_ovoo_b)
               temp_2 = np.einsum('i,bjli->jbl',r_b, v2e_vooo_ab)
              
               temp  = np.einsum('jbl,klab->ajk',temp_1,t2_1_b,optimize=True)
               temp += np.einsum('jbl,lkba->ajk',temp_2,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] -= temp[:,ij_ind_b[0],ij_ind_b[1] ].reshape(-1)
              
               temp  = -np.einsum('i,jbli->jbl',r_b,v2e_ovoo_ab,optimize=True)
               temp_1 = -np.einsum('jbl,lkab->ajk',temp,t2_1_ab,optimize=True)
               s[s_aba:f_aba] -= temp_1.reshape(-1)

        s *= -1.0

        if direct_adc.algorithm == "cvs" or direct_adc.algorithm == "mom_conventional":
            s = cvs_projector(direct_adc, s)

        return s

    precond_ = -precond.copy()
    M_ij_a_ = -M_ij_a.copy()
    M_ij_b_ = -M_ij_b.copy()

    return sigma_, precond_, (M_ij_a_, M_ij_b_)

#################################################
##### Precompute Mab block for EA ###############
#################################################
#@profile
def get_Mab(direct_adc,t_amp):

    method = direct_adc.method

    t2_1, t2_2, t1_2, t1_3 = t_amp

    t2_1_a, t2_1_ab, t2_1_b = t2_1
    t1_2_a, t1_2_b = t1_2

    nocc_a = direct_adc.nocc_a
    nocc_b = direct_adc.nocc_b
    nvir_a = direct_adc.nvir_a
    nvir_b = direct_adc.nvir_b

    e_occ_a = direct_adc.mo_energy_a[:nocc_a]
    e_occ_b = direct_adc.mo_energy_b[:nocc_b]
    e_vir_a = direct_adc.mo_energy_a[nocc_a:]
    e_vir_b = direct_adc.mo_energy_b[nocc_b:]

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    v2e_oovv_a,v2e_oovv_ab,v2e_oovv_b = direct_adc.v2e.oovv
    v2e_ooov_a,v2e_ooov_ab,v2e_ooov_b = direct_adc.v2e.ooov
    v2e_oooo_a,v2e_oooo_ab,v2e_oooo_b = direct_adc.v2e.oooo
    v2e_ovoo_a,v2e_ovoo_ab,v2e_ovoo_b = direct_adc.v2e.ovoo
    v2e_ovov_a,v2e_ovov_ab,v2e_ovov_b = direct_adc.v2e.ovov
    v2e_vvoo_a,v2e_vvoo_ab,v2e_vvoo_b = direct_adc.v2e.vvoo
    v2e_vvvv_a,v2e_vvvv_ab,v2e_vvvv_b = direct_adc.v2e.vvvv
    v2e_voov_a,v2e_voov_ab,v2e_voov_b = direct_adc.v2e.voov
    v2e_ovvo_a,v2e_ovvo_ab,v2e_ovvo_b = direct_adc.v2e.ovvo
    v2e_vovo_a,v2e_vovo_ab,v2e_vovo_b = direct_adc.v2e.vovo
    v2e_vvvo_a,v2e_vvvo_ab,v2e_vvvo_b = direct_adc.v2e.vvvo
    v2e_vovv_a,v2e_vovv_ab,v2e_vovv_b = direct_adc.v2e.vovv
    v2e_oovo_a,v2e_oovo_ab,v2e_oovo_b = direct_adc.v2e.oovo
    v2e_ovvv_a,v2e_ovvv_ab,v2e_ovvv_b = direct_adc.v2e.ovvv
    v2e_vvov_a,v2e_vvov_ab,v2e_vvov_b = direct_adc.v2e.vvov

    # a-b block
    # Zeroth-order terms
    M_ab_a = np.einsum('ab,a->ab', idn_vir_a, e_vir_a)
    M_ab_b = np.einsum('ab,a->ab', idn_vir_b, e_vir_b)

   # Second-order terms

    M_ab_a +=  np.einsum('l,lmad,lmbd->ab',e_occ_a,t2_1_a, t2_1_a)
    M_ab_a +=  np.einsum('l,lmad,lmbd->ab',e_occ_a,t2_1_ab, t2_1_ab)
    M_ab_a +=  np.einsum('l,mlad,mlbd->ab',e_occ_b,t2_1_ab, t2_1_ab)

    M_ab_b +=  np.einsum('l,lmad,lmbd->ab',e_occ_b,t2_1_b, t2_1_b)
    M_ab_b +=  np.einsum('l,mlda,mldb->ab',e_occ_b,t2_1_ab, t2_1_ab)
    M_ab_b +=  np.einsum('l,lmda,lmdb->ab',e_occ_a,t2_1_ab, t2_1_ab)

    M_ab_a -= 0.5 *  np.einsum('d,lmad,lmbd->ab',e_vir_a,t2_1_a, t2_1_a)
    M_ab_a -= 0.5 *  np.einsum('d,lmad,lmbd->ab',e_vir_b,t2_1_ab, t2_1_ab)
    M_ab_a -= 0.5 *  np.einsum('d,mlad,mlbd->ab',e_vir_b,t2_1_ab, t2_1_ab)

    M_ab_b -= 0.5 *  np.einsum('d,lmad,lmbd->ab',e_vir_b,t2_1_b, t2_1_b)
    M_ab_b -= 0.5 *  np.einsum('d,mlda,mldb->ab',e_vir_a,t2_1_ab, t2_1_ab)
    M_ab_b -= 0.5 *  np.einsum('d,lmda,lmdb->ab',e_vir_a,t2_1_ab, t2_1_ab)

    M_ab_a -= 0.25 *  np.einsum('a,lmad,lmbd->ab',e_vir_a,t2_1_a, t2_1_a)
    M_ab_a -= 0.25 *  np.einsum('a,lmad,lmbd->ab',e_vir_a,t2_1_ab, t2_1_ab)
    M_ab_a -= 0.25 *  np.einsum('a,mlad,mlbd->ab',e_vir_a,t2_1_ab, t2_1_ab)

    M_ab_b -= 0.25 *  np.einsum('a,lmad,lmbd->ab',e_vir_b,t2_1_b, t2_1_b)
    M_ab_b -= 0.25 *  np.einsum('a,mlda,mldb->ab',e_vir_b,t2_1_ab, t2_1_ab)
    M_ab_b -= 0.25 *  np.einsum('a,lmda,lmdb->ab',e_vir_b,t2_1_ab, t2_1_ab)

    M_ab_a -= 0.25 *  np.einsum('b,lmad,lmbd->ab',e_vir_a,t2_1_a, t2_1_a)
    M_ab_a -= 0.25 *  np.einsum('b,lmad,lmbd->ab',e_vir_a,t2_1_ab, t2_1_ab)
    M_ab_a -= 0.25 *  np.einsum('b,mlad,mlbd->ab',e_vir_a,t2_1_ab, t2_1_ab)

    M_ab_b -= 0.25 *  np.einsum('b,lmad,lmbd->ab',e_vir_b,t2_1_b, t2_1_b)
    M_ab_b -= 0.25 *  np.einsum('b,mlda,mldb->ab',e_vir_b,t2_1_ab, t2_1_ab)
    M_ab_b -= 0.25 *  np.einsum('b,lmda,lmdb->ab',e_vir_b,t2_1_ab, t2_1_ab)

    M_ab_a -= 0.5 *  np.einsum('lmad,lmbd->ab',t2_1_a, v2e_oovv_a)
    M_ab_a -=        np.einsum('lmad,lmbd->ab',t2_1_ab, v2e_oovv_ab)

    M_ab_b -= 0.5 *  np.einsum('lmad,lmbd->ab',t2_1_b, v2e_oovv_b)
    M_ab_b -=        np.einsum('mlda,mldb->ab',t2_1_ab, v2e_oovv_ab)

    M_ab_a -= 0.5 *  np.einsum('lmbd,lmad->ab',t2_1_a, v2e_oovv_a)
    M_ab_a -=        np.einsum('lmbd,lmad->ab',t2_1_ab, v2e_oovv_ab)

    M_ab_b -= 0.5 *  np.einsum('lmbd,lmad->ab',t2_1_b, v2e_oovv_b)
    M_ab_b -=        np.einsum('mldb,mlda->ab',t2_1_ab, v2e_oovv_ab)

    #Third-order terms
    
    if(method =='adc(3)'):

        t2_2_a, t2_2_ab, t2_2_b = t2_2

        M_ab_a +=  np.einsum('ld,albd->ab',t1_2_a, v2e_vovv_a)
        M_ab_a +=  np.einsum('ld,albd->ab',t1_2_b, v2e_vovv_ab)

        M_ab_b +=  np.einsum('ld,albd->ab',t1_2_b, v2e_vovv_b)
        M_ab_b +=  np.einsum('ld,ladb->ab',t1_2_a, v2e_ovvv_ab)

        M_ab_a += np.einsum('ld,adbl->ab',t1_2_a, v2e_vvvo_a)
        M_ab_a += np.einsum('ld,adbl->ab',t1_2_b, v2e_vvvo_ab)

        M_ab_b += np.einsum('ld,adbl->ab',t1_2_b, v2e_vvvo_b)
        M_ab_b += np.einsum('ld,dalb->ab',t1_2_a, v2e_vvov_ab)

        M_ab_a -=0.5* np.einsum('lmbd,lmad->ab',t2_2_a,v2e_oovv_a)
        M_ab_a -= np.einsum('lmbd,lmad->ab',t2_2_ab,v2e_oovv_ab)

        M_ab_b -=0.5* np.einsum('lmbd,lmad->ab',t2_2_b,v2e_oovv_b)
        M_ab_b -= np.einsum('mldb,mlda->ab',t2_2_ab,v2e_oovv_ab)

        M_ab_a -=0.5* np.einsum('lmad,lmbd->ab',t2_2_a,v2e_oovv_a)
        M_ab_a -= np.einsum('lmad,lmbd->ab',t2_2_ab,v2e_oovv_ab)

        M_ab_b -=0.5* np.einsum('lmad,lmbd->ab',t2_2_b,v2e_oovv_b)
        M_ab_b -= np.einsum('mlda,mldb->ab',t2_2_ab,v2e_oovv_ab)

        M_ab_a += np.einsum('l,lmbd,lmad->ab',e_occ_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a += np.einsum('l,lmbd,lmad->ab',e_occ_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a += np.einsum('l,mlbd,mlad->ab',e_occ_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b += np.einsum('l,lmbd,lmad->ab',e_occ_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b += np.einsum('l,mldb,mlda->ab',e_occ_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b += np.einsum('l,lmdb,lmda->ab',e_occ_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a += np.einsum('l,lmad,lmbd->ab',e_occ_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a += np.einsum('l,lmad,lmbd->ab',e_occ_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a += np.einsum('l,mlad,mlbd->ab',e_occ_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b += np.einsum('l,lmad,lmbd->ab',e_occ_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b += np.einsum('l,mlda,mldb->ab',e_occ_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b += np.einsum('l,lmda,lmdb->ab',e_occ_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= 0.5*np.einsum('d,lmbd,lmad->ab', e_vir_a, t2_1_a ,t2_2_a, optimize=True)
        M_ab_a -= 0.5*np.einsum('d,lmbd,lmad->ab', e_vir_b, t2_1_ab ,t2_2_ab, optimize=True)
        M_ab_a -= 0.5*np.einsum('d,mlbd,mlad->ab', e_vir_b, t2_1_ab ,t2_2_ab, optimize=True)

        M_ab_b -= 0.5*np.einsum('d,lmbd,lmad->ab', e_vir_b, t2_1_b ,t2_2_b, optimize=True)
        M_ab_b -= 0.5*np.einsum('d,mldb,mlda->ab', e_vir_a, t2_1_ab ,t2_2_ab, optimize=True)
        M_ab_b -= 0.5*np.einsum('d,lmdb,lmda->ab', e_vir_a, t2_1_ab ,t2_2_ab, optimize=True)

        M_ab_a -= 0.5*np.einsum('d,lmad,lmbd->ab', e_vir_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a -= 0.5*np.einsum('d,lmad,lmbd->ab', e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a -= 0.5*np.einsum('d,mlad,mlbd->ab', e_vir_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b -= 0.5*np.einsum('d,lmad,lmbd->ab', e_vir_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b -= 0.5*np.einsum('d,mlda,mldb->ab', e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b -= 0.5*np.einsum('d,lmda,lmdb->ab', e_vir_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= 0.25*np.einsum('a,lmbd,lmad->ab',e_vir_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a -= 0.25*np.einsum('a,lmbd,lmad->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a -= 0.25*np.einsum('a,mlbd,mlad->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b -= 0.25*np.einsum('a,lmbd,lmad->ab',e_vir_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b -= 0.25*np.einsum('a,mldb,mlda->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b -= 0.25*np.einsum('a,lmdb,lmda->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= 0.25*np.einsum('a,lmad,lmbd->ab',e_vir_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a -= 0.25*np.einsum('a,lmad,lmbd->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a -= 0.25*np.einsum('a,mlad,mlbd->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b -= 0.25*np.einsum('a,lmad,lmbd->ab',e_vir_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b -= 0.25*np.einsum('a,mlda,mldb->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b -= 0.25*np.einsum('a,lmda,lmdb->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= 0.25*np.einsum('b,lmbd,lmad->ab',e_vir_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a -= 0.25*np.einsum('b,lmbd,lmad->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a -= 0.25*np.einsum('b,mlbd,mlad->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b -= 0.25*np.einsum('b,lmbd,lmad->ab',e_vir_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b -= 0.25*np.einsum('b,mldb,mlda->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b -= 0.25*np.einsum('b,lmdb,lmda->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= 0.25*np.einsum('b,lmad,lmbd->ab',e_vir_a, t2_1_a, t2_2_a, optimize=True)
        M_ab_a -= 0.25*np.einsum('b,lmad,lmbd->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_a -= 0.25*np.einsum('b,mlad,mlbd->ab',e_vir_a, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_b -= 0.25*np.einsum('b,lmad,lmbd->ab',e_vir_b, t2_1_b, t2_2_b, optimize=True)
        M_ab_b -= 0.25*np.einsum('b,mlda,mldb->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)
        M_ab_b -= 0.25*np.einsum('b,lmda,lmdb->ab',e_vir_b, t2_1_ab, t2_2_ab, optimize=True)

        M_ab_a -= np.einsum('lned,mlbd,anem->ab',t2_1_a, t2_1_a, v2e_vovo_a, optimize=True)
        M_ab_a += np.einsum('nled,mlbd,anem->ab',t2_1_ab, t2_1_ab, v2e_vovo_a, optimize=True)
        M_ab_a -= np.einsum('lnde,mlbd,anme->ab',t2_1_ab, t2_1_a, v2e_voov_ab, optimize=True)
        M_ab_a += np.einsum('lned,mlbd,anme->ab',t2_1_b, t2_1_ab, v2e_voov_ab, optimize=True)
        M_ab_a += np.einsum('lned,lmbd,anem->ab',t2_1_ab, t2_1_ab, v2e_vovo_ab, optimize=True)

        M_ab_b -= np.einsum('lned,mlbd,anem->ab',t2_1_b, t2_1_b, v2e_vovo_b, optimize=True)
        M_ab_b += np.einsum('lnde,lmdb,anem->ab',t2_1_ab, t2_1_ab, v2e_vovo_b, optimize=True)
        M_ab_b -= np.einsum('nled,mlbd,naem->ab',t2_1_ab, t2_1_b, v2e_ovvo_ab, optimize=True)
        M_ab_b += np.einsum('lned,lmdb,naem->ab',t2_1_a, t2_1_ab, v2e_ovvo_ab, optimize=True)
        M_ab_b += np.einsum('nlde,mldb,name->ab',t2_1_ab, t2_1_ab, v2e_ovov_ab, optimize=True)

        M_ab_a -= np.einsum('mled,lnad,enbm->ab',t2_1_a, t2_1_a, v2e_vovo_a, optimize=True)
        M_ab_a -= np.einsum('mled,nlad,nebm->ab',t2_1_b, t2_1_ab, v2e_ovvo_ab, optimize=True)
        M_ab_a += np.einsum('mled,nlad,enbm->ab',t2_1_ab, t2_1_ab, v2e_vovo_a, optimize=True)
        M_ab_a += np.einsum('lmde,lnad,nebm->ab',t2_1_ab, t2_1_a, v2e_ovvo_ab, optimize=True)
        M_ab_a += np.einsum('lmed,lnad,enbm->ab',t2_1_ab, t2_1_ab, v2e_vovo_ab, optimize=True)
 
        M_ab_b -= np.einsum('mled,lnad,enbm->ab',t2_1_b, t2_1_b, v2e_vovo_b, optimize=True)
        M_ab_b -= np.einsum('mled,lnda,enmb->ab',t2_1_a, t2_1_ab, v2e_voov_ab, optimize=True)
        M_ab_b += np.einsum('lmde,lnda,enbm->ab',t2_1_ab, t2_1_ab, v2e_vovo_b, optimize=True)
        M_ab_b += np.einsum('mled,lnad,enmb->ab',t2_1_ab, t2_1_b, v2e_voov_ab, optimize=True)
        M_ab_b += np.einsum('mlde,nlda,nemb->ab',t2_1_ab, t2_1_ab, v2e_ovov_ab, optimize=True)
 
        M_ab_a -= np.einsum('mlbd,lnae,dnem->ab',t2_1_a, t2_1_a, v2e_vovo_a, optimize=True)
        M_ab_a += np.einsum('lmbd,lnae,dnem->ab',t2_1_ab, t2_1_ab, v2e_vovo_b, optimize=True)
        M_ab_a += np.einsum('mlbd,lnae,dnme->ab',t2_1_a, t2_1_ab, v2e_voov_ab, optimize=True)
        M_ab_a -= np.einsum('lmbd,lnae,ndem->ab',t2_1_ab, t2_1_a, v2e_ovvo_ab, optimize=True)
        M_ab_a += np.einsum('mlbd,nlae,ndme->ab',t2_1_ab, t2_1_ab, v2e_ovov_ab, optimize=True)
 
        M_ab_b -= np.einsum('mlbd,lnae,dnem->ab',t2_1_b, t2_1_b, v2e_vovo_b, optimize=True)
        M_ab_b += np.einsum('mldb,nlea,dnem->ab',t2_1_ab, t2_1_ab, v2e_vovo_a, optimize=True)
        M_ab_b += np.einsum('mlbd,nlea,ndem->ab',t2_1_b, t2_1_ab, v2e_ovvo_ab, optimize=True)
        M_ab_b -= np.einsum('mldb,lnae,dnme->ab',t2_1_ab, t2_1_b, v2e_voov_ab, optimize=True)
        M_ab_b += np.einsum('lmdb,lnea,dnem->ab',t2_1_ab, t2_1_ab, v2e_vovo_ab, optimize=True)
 
        M_ab_a -= 0.25*np.einsum('mlef,mlbd,adef->ab',t2_1_a, t2_1_a, v2e_vvvv_a, optimize=True)
        #temp = t2_1_a.reshape(nocc_a*nocc_a,-1)
        #temp_1 = v2e_vvvv_a.reshape(nvir_a,-1)
        #int_1 = np.dot(temp.T,temp).reshape(nvir_a,-1)
        #M_ab_a -= 0.25*np.dot(temp_1,int_1.T) 
        M_ab_a -= np.einsum('mlef,mlbd,adef->ab',t2_1_ab, t2_1_ab, v2e_vvvv_ab, optimize=True)
 
        M_ab_b -= 0.25*np.einsum('mlef,mlbd,adef->ab',t2_1_b, t2_1_b, v2e_vvvv_b, optimize=True)
        M_ab_b -= np.einsum('mlef,mldb,daef->ab',t2_1_ab, t2_1_ab, v2e_vvvv_ab, optimize=True)
 
        M_ab_a -= 0.25*np.einsum('mled,mlaf,edbf->ab',t2_1_a, t2_1_a, v2e_vvvv_a, optimize=True)
        M_ab_a -= np.einsum('mled,mlaf,edbf->ab',t2_1_ab, t2_1_ab, v2e_vvvv_ab, optimize=True)
 
        M_ab_b -= 0.25*np.einsum('mled,mlaf,edbf->ab',t2_1_b, t2_1_b, v2e_vvvv_b, optimize=True)
        M_ab_b -= np.einsum('mled,mlfa,edfb->ab',t2_1_ab, t2_1_ab, v2e_vvvv_ab, optimize=True)
 
        M_ab_a -= 0.25*np.einsum('mlbd,noad,noml->ab',t2_1_a, t2_1_a, v2e_oooo_a, optimize=True)
        M_ab_a -= np.einsum('mlbd,noad,noml->ab',t2_1_ab, t2_1_ab, v2e_oooo_ab, optimize=True)
 
        M_ab_b -= 0.25*np.einsum('mlbd,noad,noml->ab',t2_1_b, t2_1_b, v2e_oooo_b, optimize=True)
        M_ab_b -= np.einsum('lmdb,onda,onlm->ab',t2_1_ab, t2_1_ab, v2e_oooo_ab, optimize=True)
 
        M_ab_a += 0.5*np.einsum('lned,mled,anbm->ab',t2_1_a, t2_1_a, v2e_vovo_a, optimize=True)
        M_ab_a += 0.5*np.einsum('lned,mled,anbm->ab',t2_1_b, t2_1_b, v2e_vovo_ab, optimize=True)
        M_ab_a -= np.einsum('lned,lmed,anbm->ab',t2_1_ab, t2_1_ab, v2e_vovo_ab, optimize=True)
        M_ab_a -= np.einsum('nled,mled,anbm->ab',t2_1_ab, t2_1_ab, v2e_vovo_a, optimize=True)
 
        M_ab_b += 0.5*np.einsum('lned,mled,anbm->ab',t2_1_b, t2_1_b, v2e_vovo_b, optimize=True)
        M_ab_b += 0.5*np.einsum('lned,mled,namb->ab',t2_1_a, t2_1_a, v2e_ovov_ab, optimize=True)
        M_ab_b -= np.einsum('nled,mled,namb->ab',t2_1_ab, t2_1_ab, v2e_ovov_ab, optimize=True)
        M_ab_b -= np.einsum('lned,lmed,anbm->ab',t2_1_ab, t2_1_ab, v2e_vovo_b, optimize=True)
 
        M_ab_a -= 0.5*np.einsum('mldf,mled,aebf->ab',t2_1_a, t2_1_a, v2e_vvvv_a, optimize=True)
        M_ab_a -= 0.5*np.einsum('mldf,mled,aebf->ab',t2_1_b, t2_1_b, v2e_vvvv_ab, optimize=True)
        M_ab_a += np.einsum('mldf,mlde,aebf->ab',t2_1_ab, t2_1_ab, v2e_vvvv_ab, optimize=True)
        M_ab_a += np.einsum('mlfd,mled,aebf->ab',t2_1_ab, t2_1_ab, v2e_vvvv_a, optimize=True)
        
        M_ab_b -= 0.5*np.einsum('mldf,mled,aebf->ab',t2_1_b, t2_1_b, v2e_vvvv_b, optimize=True)
        M_ab_b -= 0.5*np.einsum('mldf,mled,eafb->ab',t2_1_a, t2_1_a, v2e_vvvv_ab, optimize=True)
        M_ab_b += np.einsum('mlfd,mled,eafb->ab',t2_1_ab, t2_1_ab, v2e_vvvv_ab, optimize=True)
        M_ab_b += np.einsum('mldf,mlde,aebf->ab',t2_1_ab, t2_1_ab, v2e_vvvv_b, optimize=True)

    M_ab = (M_ab_a, M_ab_b)

    return M_ab

###########################################
# Compute sigma vector for EA #
###########################################

def define_H_ea(direct_adc,t_amp):

    method = direct_adc.method

    t2_1, t2_2, t1_2, t1_3 = t_amp

    t2_1_a, t2_1_ab, t2_1_b = t2_1
    t1_2_a, t1_2_b = t1_2

    nocc_a = direct_adc.nocc_a
    nocc_b = direct_adc.nocc_b
    nvir_a = direct_adc.nvir_a
    nvir_b = direct_adc.nvir_b
    
    ab_ind_a = np.tril_indices(nvir_a, k=-1)
    ab_ind_b = np.tril_indices(nvir_b, k=-1)

    n_singles_a = nvir_a
    n_singles_b = nvir_b
    n_doubles_aaa = nvir_a * (nvir_a - 1) * nocc_a // 2
    n_doubles_bab = nocc_b * nvir_a * nvir_b
    n_doubles_aba = nocc_a * nvir_b * nvir_a
    n_doubles_bbb = nvir_b * (nvir_b - 1) * nocc_b // 2

    dim = n_singles_a + n_singles_b + n_doubles_aaa + n_doubles_bab + n_doubles_aba + n_doubles_bbb

    e_occ_a = direct_adc.mo_energy_a[:nocc_a]
    e_occ_b = direct_adc.mo_energy_b[:nocc_b]
    e_vir_a = direct_adc.mo_energy_a[nocc_a:]
    e_vir_b = direct_adc.mo_energy_b[nocc_b:]

    idn_occ_a = np.identity(nocc_a)
    idn_occ_b = np.identity(nocc_b)
    idn_vir_a = np.identity(nvir_a)
    idn_vir_b = np.identity(nvir_b)

    v2e_oovv_a,v2e_oovv_ab,v2e_oovv_b = direct_adc.v2e.oovv
    v2e_ooov_a,v2e_ooov_ab,v2e_ooov_b = direct_adc.v2e.ooov
    v2e_oooo_a,v2e_oooo_ab,v2e_oooo_b = direct_adc.v2e.oooo
    v2e_ovoo_a,v2e_ovoo_ab,v2e_ovoo_b = direct_adc.v2e.ovoo
    v2e_ovov_a,v2e_ovov_ab,v2e_ovov_b = direct_adc.v2e.ovov
    v2e_vvoo_a,v2e_vvoo_ab,v2e_vvoo_b = direct_adc.v2e.vvoo
    v2e_vvvv_a,v2e_vvvv_ab,v2e_vvvv_b = direct_adc.v2e.vvvv
    v2e_voov_a,v2e_voov_ab,v2e_voov_b = direct_adc.v2e.voov
    v2e_ovvo_a,v2e_ovvo_ab,v2e_ovvo_b = direct_adc.v2e.ovvo
    v2e_vovo_a,v2e_vovo_ab,v2e_vovo_b = direct_adc.v2e.vovo
    v2e_vvvo_a,v2e_vvvo_ab,v2e_vvvo_b = direct_adc.v2e.vvvo
    v2e_vovv_a,v2e_vovv_ab,v2e_vovv_b = direct_adc.v2e.vovv
    v2e_oovo_a,v2e_oovo_ab,v2e_oovo_b = direct_adc.v2e.oovo
    v2e_ovvv_a,v2e_ovvv_ab,v2e_ovvv_b = direct_adc.v2e.ovvv

    v2e_vovv_1_a = v2e_vovv_a.copy()[:,:,ab_ind_a[0],ab_ind_a[1]].reshape(nvir_a,-1)
    v2e_vovv_1_b = v2e_vovv_b.copy()[:,:,ab_ind_b[0],ab_ind_b[1]].reshape(nvir_b,-1)
    
    v2e_vovv_2_a = v2e_vovv_a.copy()[:,:,ab_ind_a[0],ab_ind_a[1]]
    v2e_vovv_2_b = v2e_vovv_b.copy()[:,:,ab_ind_b[0],ab_ind_b[1]]

    d_i_a = e_occ_a[:,None]
    d_ab_a = e_vir_a[:,None] + e_vir_a
    D_n_a = -d_i_a + d_ab_a.reshape(-1)
    D_n_a = D_n_a.reshape((nocc_a,nvir_a,nvir_a))
    D_iab_a = D_n_a.copy()[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

    d_i_b = e_occ_b[:,None]
    d_ab_b = e_vir_b[:,None] + e_vir_b
    D_n_b = -d_i_b + d_ab_b.reshape(-1)
    D_n_b = D_n_b.reshape((nocc_b,nvir_b,nvir_b))
    D_iab_b = D_n_b.copy()[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

    d_ab_ab = e_vir_a[:,None] + e_vir_b
    d_i_b = e_occ_b[:,None]
    D_n_bab = -d_i_b + d_ab_ab.reshape(-1)
    D_iab_bab = D_n_bab.reshape(-1)

    d_ab_ab = e_vir_b[:,None] + e_vir_a
    d_i_a = e_occ_a[:,None]
    D_n_aba = -d_i_a + d_ab_ab.reshape(-1)
    D_iab_aba = D_n_aba.reshape(-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb
    
    M_ab_a, M_ab_b = get_Mab(direct_adc,t_amp)

    precond = np.zeros(dim)

    # Compute precond in p1-p1 block
    M_ab_a_diag = np.diagonal(M_ab_a)
    M_ab_b_diag = np.diagonal(M_ab_b)

    precond[s_a:f_a] = M_ab_a_diag.copy()
    precond[s_b:f_b] = M_ab_b_diag.copy()

    # Compute precond in 2p1h-2p1h block
  
    precond[s_aaa:f_aaa] = D_iab_a
    precond[s_bab:f_bab] = D_iab_bab
    precond[s_aba:f_aba] = D_iab_aba
    precond[s_bbb:f_bbb] = D_iab_b

    #Calculate sigma vector
    #@profile
    def sigma_(r):

        s = None
        if direct_adc.algorithm == "dynamical":
            z = np.zeros((dim))
            s = np.array(z,dtype = complex)
            s.imag = z
        else:
            s = np.zeros((dim))

        r_a = r[s_a:f_a]
        r_b = r[s_b:f_b]

        r_aaa = r[s_aaa:f_aaa]
        r_bab = r[s_bab:f_bab]
        r_aba = r[s_aba:f_aba]
        r_bbb = r[s_bbb:f_bbb]

        r_aba = r_aba.reshape(nocc_a,nvir_b,nvir_a)
        r_bab = r_bab.reshape(nocc_b,nvir_a,nvir_b)

############ ADC(2) ab block ############################
        
        s[s_a:f_a] = np.einsum('ab,b->a',M_ab_a,r_a)
        s[s_b:f_b] = np.einsum('ab,b->a',M_ab_b,r_b)
  
############ ADC(2) a - ibc block #########################

        s[s_a:f_a] += np.einsum('ap,p->a',v2e_vovv_1_a, r_aaa, optimize = True)
        s[s_a:f_a] += np.einsum('aibc,ibc->a', v2e_vovv_ab, r_bab, optimize = True)

        s[s_b:f_b] += np.einsum('ap,p->a', v2e_vovv_1_b, r_bbb, optimize = True)
        s[s_b:f_b] += np.einsum('iacb,ibc->a', v2e_ovvv_ab, r_aba, optimize = True)

############### ADC(2) ibc - a block ############################

        s[s_aaa:f_aaa] += np.einsum('aip,a->ip', v2e_vovv_2_a, r_a, optimize = True).reshape(-1)
        #temp = v2e_vovv_2_a.reshape(nvir_a,-1)
        #s[s_aaa:f_aaa] += np.dot(r_a,temp)
        s[s_bab:f_bab] += np.einsum('aibc,a->ibc', v2e_vovv_ab, r_a, optimize = True).reshape(-1)

        s[s_aba:f_aba] += np.einsum('iacb,a->ibc', v2e_ovvv_ab, r_b, optimize = True).reshape(-1)
        s[s_bbb:f_bbb] += np.einsum('aip,a->ip', v2e_vovv_2_b, r_b, optimize = True).reshape(-1)
        #temp = v2e_vovv_2_b.reshape(nvir_b,-1)
        #s[s_bbb:f_bbb] += np.dot(r_b,temp)

################ ADC(2) iab - jcd block ############################ 
        s[s_aaa:f_aaa] += D_iab_a * r_aaa
        s[s_bab:f_bab] += D_iab_bab * r_bab.reshape(-1)
        s[s_aba:f_aba] += D_iab_aba * r_aba.reshape(-1)
        s[s_bbb:f_bbb] += D_iab_b * r_bbb

############### ADC(3) iab - jcd block ############################

        if (method == "adc(2)-e" or method == "adc(3)"):
       	
               t2_2_a, t2_2_ab, t2_2_b = t2_2


               r_aaa = r_aaa.reshape(nocc_a,-1)
               r_bbb = r_bbb.reshape(nocc_b,-1)

               r_aaa_u = None
               if direct_adc.algorithm == "dynamical":
                   r_aaa_u = np.zeros((nocc_a,nvir_a,nvir_a),dtype=complex)
               else:
                   r_aaa_u = np.zeros((nocc_a,nvir_a,nvir_a))
               r_aaa_u[:,ab_ind_a[0],ab_ind_a[1]]= r_aaa.copy()
               r_aaa_u[:,ab_ind_a[1],ab_ind_a[0]]= -r_aaa.copy()

               r_bbb_u = None
               if direct_adc.algorithm == "dynamical":
                   r_bbb_u = np.zeros((nocc_b,nvir_b,nvir_b),dtype=complex)
               else:
                   r_bbb_u = np.zeros((nocc_b,nvir_b,nvir_b))
               r_bbb_u[:,ab_ind_b[0],ab_ind_b[1]]= r_bbb.copy()
               r_bbb_u[:,ab_ind_b[1],ab_ind_b[0]]= -r_bbb.copy()

               #temp = 0.5*np.einsum('yxwz,izw->ixy',v2e_vvvv_a,r_aaa_u ,optimize = True)
               #####temp = -0.5*np.einsum('yxzw,izw->ixy',v2e_vvvv_a,r_aaa_u )
               #s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

               #temp = v2e_vvvv_a[ab_ind_a[0],ab_ind_a[1],:,:]
               #temp = temp.reshape(-1,nvir_a*nvir_a)
               #r_aaa_t = r_aaa_u.reshape(nocc_a,-1)
               #s[s_aaa:f_aaa] += 0.5*np.dot(r_aaa_t,temp.T).reshape(-1)

               temp = v2e_vvvv_a[:].reshape(nvir_a*nvir_a,nvir_a*nvir_a)
               r_aaa_t = r_aaa_u.reshape(nocc_a,-1)
               temp_1 = np.dot(r_aaa_t,temp.T).reshape(nocc_a,nvir_a,nvir_a)
               del temp
               temp_1 = temp_1[:,ab_ind_a[0],ab_ind_a[1]]               
               s[s_aaa:f_aaa] += 0.5*temp_1.reshape(-1)

               temp = v2e_vvvv_b[:].reshape(nvir_b*nvir_b,nvir_b*nvir_b)
               r_bbb_t = r_bbb_u.reshape(nocc_b,-1)
               temp_1 = np.dot(r_bbb_t,temp.T).reshape(nocc_b,nvir_b,nvir_b)
               del temp
               temp_1 = temp_1[:,ab_ind_b[0],ab_ind_b[1]]               
               s[s_bbb:f_bbb] += 0.5*temp_1.reshape(-1)

               #temp = v2e_vvvv_b[ab_ind_b[0],ab_ind_b[1],:,:]
               #temp = temp.reshape(-1,nvir_b*nvir_b)
               #r_bbb_t = r_bbb_u.reshape(nocc_b,-1)
               #s[s_bbb:f_bbb] += 0.5*np.dot(r_bbb_t,temp.T).reshape(-1)

               #temp = 0.5*np.einsum('yxwz,izw->ixy',v2e_vvvv_b,r_bbb_u,optimize = True)
               ########temp = -0.5*np.einsum('yxzw,izw->ixy',v2e_vvvv_b,r_bbb_u)
               #s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

               #s[s_bab:f_bab] += np.einsum('xyzw,izw->ixy',v2e_vvvv_ab,r_bab,optimize = True).reshape(-1)
               #s[s_bab:f_bab] += np.einsum('xyzw,izw->ixy',v2e_vvvv_ab,r_bab).reshape(-1)
               temp = v2e_vvvv_ab[:].reshape(nvir_a*nvir_b,nvir_a*nvir_b)
               r_bab_t = r_bab.reshape(nocc_b,-1)
               s[s_bab:f_bab] += np.dot(r_bab_t,temp.T).reshape(-1)
               del temp

               #s[s_aba:f_aba] += np.einsum('yxwz,izw->ixy',v2e_vvvv_ab,r_aba,optimize = True).reshape(-1)
               #temp = v2e_vvvv_ab.transpose(3,2,1,0) 
               #temp = temp.reshape(nvir_a*nvir_b,nvir_a*nvir_b)
               #r_aba_t = r_aba.reshape(nocc_a,-1)
               #s[s_aba:f_aba] += np.dot(r_aba_t,temp).reshape(-1)

               temp = v2e_vvvv_ab[:].reshape(nvir_a*nvir_b,nvir_a*nvir_b)
               r_aba_t = r_aba.transpose(0,2,1).reshape(nocc_a,-1)
               temp_1 = np.dot(r_aba_t,temp.T).reshape(nocc_a, nvir_a,nvir_b)
               s[s_aba:f_aba] += temp_1.transpose(0,2,1).copy().reshape(-1)
               del temp

               temp = 0.5*np.einsum('yjzi,jzx->ixy',v2e_vovo_a,r_aaa_u,optimize = True)
               temp +=0.5*np.einsum('yjiz,jxz->ixy',v2e_voov_ab,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] -= 0.5*np.einsum('jyzi,jzx->ixy',v2e_ovvo_ab,r_aaa_u,optimize = True).reshape(-1)
               s[s_bab:f_bab] -= 0.5*np.einsum('yjzi,jxz->ixy',v2e_vovo_b,r_bab,optimize = True).reshape(-1)

               temp = 0.5*np.einsum('yjzi,jzx->ixy',v2e_vovo_b,r_bbb_u,optimize = True)
               temp +=0.5* np.einsum('jyzi,jxz->ixy',v2e_ovvo_ab,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

               s[s_aba:f_aba] -= 0.5*np.einsum('yjzi,jxz->ixy',v2e_vovo_a,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] -= 0.5*np.einsum('yjiz,jzx->ixy',v2e_voov_ab,r_bbb_u,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('xjzi,jzy->ixy',v2e_vovo_a,r_aaa_u,optimize = True)
               temp -= 0.5*np.einsum('xjiz,jyz->ixy',v2e_voov_ab,r_bab,optimize = True)
               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] -=  0.5*np.einsum('xjzi,jzy->ixy',v2e_vovo_ab,r_bab,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('xjzi,jzy->ixy',v2e_vovo_b,r_bbb_u,optimize = True)
               temp -= 0.5*np.einsum('jxzi,jyz->ixy',v2e_ovvo_ab,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

               s[s_aba:f_aba] -= 0.5*np.einsum('jxiz,jzy->ixy',v2e_ovov_ab,r_aba,optimize = True).reshape(-1)

               temp = 0.5*np.einsum('xjwi,jyw->ixy',v2e_vovo_a,r_aaa_u,optimize = True)
               temp -= 0.5*np.einsum('xjiw,jyw->ixy',v2e_voov_ab,r_bab,optimize = True)

               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] -= 0.5*np.einsum('xjwi,jwy->ixy',v2e_vovo_ab,r_bab,optimize = True).reshape(-1)

               temp = 0.5*np.einsum('xjwi,jyw->ixy',v2e_vovo_b,r_bbb_u,optimize = True)
               temp -= 0.5*np.einsum('jxwi,jyw->ixy',v2e_ovvo_ab,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

               s[s_aba:f_aba] -= 0.5*np.einsum('jxiw,jwy->ixy',v2e_ovov_ab,r_aba,optimize = True).reshape(-1)
                
               temp = -0.5*np.einsum('yjwi,jxw->ixy',v2e_vovo_a,r_aaa_u,optimize = True)
               temp += 0.5*np.einsum('yjiw,jxw->ixy',v2e_voov_ab,r_bab,optimize = True)

               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1]].reshape(-1)

               s[s_bab:f_bab] -= 0.5*np.einsum('yjwi,jxw->ixy',v2e_vovo_b,r_bab,optimize = True).reshape(-1)
               s[s_bab:f_bab] += 0.5*np.einsum('jywi,jxw->ixy',v2e_ovvo_ab,r_aaa_u,optimize = True).reshape(-1)

               s[s_aba:f_aba] -= 0.5*np.einsum('yjwi,jxw->ixy',v2e_vovo_a,r_aba,optimize = True).reshape(-1)
               s[s_aba:f_aba] += 0.5*np.einsum('yjiw,jxw->ixy',v2e_voov_ab,r_bbb_u,optimize = True).reshape(-1)

               temp = -0.5*np.einsum('yjwi,jxw->ixy',v2e_vovo_b,r_bbb_u,optimize = True)
               temp += 0.5*np.einsum('jywi,jxw->ixy',v2e_ovvo_ab,r_aba,optimize = True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1]].reshape(-1)

        if (method == "adc(3)"):
      
            #print("Calculating additional terms for adc(3)")
           
############### ADC(3) a - ibc block ############################

               #temp = -0.5*np.einsum('lmwz,lmaj->ajzw',t2_1_a,v2e_oovo_a)
               #temp = temp[:,:,ab_ind_a[0],ab_ind_a[1]]
               #r_aaa = r_aaa.reshape(nocc_a,-1)
               #s[s_a:f_a] += np.einsum('ajp,jp->a',temp, r_aaa, optimize=True)

               t2_1_a_t = t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]]
               r_aaa = r_aaa.reshape(nocc_a,-1)
               temp = 0.5*np.einsum('lmp,jp->lmj',t2_1_a_t,r_aaa)
               s[s_a:f_a] += np.einsum('lmj,lmaj->a',temp, v2e_oovo_a, optimize=True)

               temp_1 = -np.einsum('lmzw,jzw->jlm',t2_1_ab,r_bab)
               s[s_a:f_a] -= np.einsum('jlm,lmaj->a',temp_1, v2e_oovo_ab, optimize=True)

               #temp = -0.5*np.einsum('lmwz,lmaj->ajzw',t2_1_b,v2e_oovo_b)
               #temp = temp[:,:,ab_ind_b[0],ab_ind_b[1]]
               #r_bbb = r_bbb.reshape(nocc_b,-1)
               #s[s_b:f_b] += np.einsum('ajp,jp->a',temp, r_bbb, optimize=True)

               t2_1_b_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]]
               r_bbb = r_bbb.reshape(nocc_b,-1)
               temp = 0.5*np.einsum('lmp,jp->lmj',t2_1_b_t,r_bbb)
               s[s_b:f_b] += np.einsum('lmj,lmaj->a',temp, v2e_oovo_b, optimize=True)

               temp_1 = -np.einsum('mlwz,jzw->jlm',t2_1_ab,r_aba)
               s[s_b:f_b] -= np.einsum('jlm,mlja->a',temp_1, v2e_ooov_ab, optimize=True)

               if direct_adc.algorithm == "dynamical":
                   r_aaa_u = np.zeros((nocc_a,nvir_a,nvir_a),dtype=complex)
               else:
                   r_aaa_u = np.zeros((nocc_a,nvir_a,nvir_a))
               r_aaa_u[:,ab_ind_a[0],ab_ind_a[1]]= r_aaa.copy()
               r_aaa_u[:,ab_ind_a[1],ab_ind_a[0]]= -r_aaa.copy()

               if direct_adc.algorithm == "dynamical":
                   r_bbb_u = np.zeros((nocc_b,nvir_b,nvir_b),dtype=complex)
               else:
                   r_bbb_u = np.zeros((nocc_b,nvir_b,nvir_b))
               r_bbb_u[:,ab_ind_b[0],ab_ind_b[1]]= r_bbb.copy()
               r_bbb_u[:,ab_ind_b[1],ab_ind_b[0]]= -r_bbb.copy()

               r_bab = r_bab.reshape(nocc_b,nvir_a,nvir_b)
               r_aba = r_aba.reshape(nocc_a,nvir_b,nvir_a)

               if direct_adc.algorithm == "dynamical":
                   temp = np.zeros_like(r_bab,dtype=complex)
               else:
                   temp = np.zeros_like(r_bab)

               temp = np.einsum('jlwd,jzw->lzd',t2_1_a,r_aaa_u,optimize=True)
               temp += np.einsum('ljdw,jzw->lzd',t2_1_ab,r_bab,optimize=True)

               if direct_adc.algorithm == "dynamical":
                   temp_1 = np.zeros_like(r_bab,dtype=complex)
               else:
                   temp_1 = np.zeros_like(r_bab)

               temp_1 = np.einsum('jlwd,jzw->lzd',t2_1_ab,r_aaa_u,optimize=True)
               temp_1 += np.einsum('jlwd,jzw->lzd',t2_1_b,r_bab,optimize=True)
               
               #temp_2 = np.einsum('ljwd,jwz->lzd',t2_1_ab,r_bab)
               
               temp_a = t2_1_ab.transpose(0,3,1,2).copy()
               temp_b = temp_a.reshape(nocc_a*nvir_b,nocc_b*nvir_a)
               r_bab_t = r_bab.reshape(nocc_b*nvir_a,-1)
               temp_c = np.dot(temp_b,r_bab_t).reshape(nocc_a,nvir_b,nvir_b)
               temp_2 = temp_c.transpose(0,2,1).copy() 

               
               s[s_a:f_a] += 0.5*np.einsum('lzd,zlad->a',temp,v2e_vovv_a,optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('lzd,zlad->a',temp_1,v2e_vovv_ab,optimize=True)
               s[s_a:f_a] -= 0.5*np.einsum('lzd,lzad->a',temp_2,v2e_ovvv_ab,optimize=True)

               if direct_adc.algorithm == "dynamical":
                   temp = np.zeros_like(r_aba,dtype=complex)
               else:
                   temp = np.zeros_like(r_aba)
               temp = np.einsum('jlwd,jzw->lzd',t2_1_b,r_bbb_u,optimize=True)
               temp += np.einsum('jlwd,jzw->lzd',t2_1_ab,r_aba,optimize=True)

               if direct_adc.algorithm == "dynamical":
                   temp_1 = np.zeros_like(r_aba,dtype=complex)
               else:
                   temp_1 = np.zeros_like(r_aba)
               temp_1 = np.einsum('ljdw,jzw->lzd',t2_1_ab,r_bbb_u,optimize=True)
               temp_1 += np.einsum('jlwd,jzw->lzd',t2_1_a,r_aba,optimize=True)
               
               temp_2 = np.einsum('jldw,jwz->lzd',t2_1_ab,r_aba,optimize=True)
               
               s[s_b:f_b] += 0.5*np.einsum('lzd,zlad->a',temp,v2e_vovv_b,optimize=True)
               #s[s_b:f_b] += 0.5*np.einsum('lzd,lzda->a',temp_1,v2e_ovvv_ab,optimize=True)
               temp_a = temp_1.reshape(-1)
               temp_b = v2e_ovvv_ab.reshape(nocc_a*nvir_b*nvir_a,-1)
               s[s_b:f_b] += 0.5*np.dot(temp_a,temp_b)
               s[s_b:f_b] -= 0.5*np.einsum('lzd,zlda->a',temp_2,v2e_vovv_ab,optimize=True)

               if direct_adc.algorithm == "dynamical":
                   temp = np.zeros_like(r_bab,dtype=complex)
               else:
                   temp = np.zeros_like(r_bab)
               temp = -np.einsum('jlzd,jwz->lwd',t2_1_a,r_aaa_u,optimize=True)
               temp += -np.einsum('ljdz,jwz->lwd',t2_1_ab,r_bab,optimize=True)
               
               if direct_adc.algorithm == "dynamical":
                   temp_1 = np.zeros_like(r_bab,dtype=complex)
               else:
                   temp_1 = np.zeros_like(r_bab)
               temp_1 = -np.einsum('jlzd,jwz->lwd',t2_1_ab,r_aaa_u,optimize=True)
               temp_1 += -np.einsum('jlzd,jwz->lwd',t2_1_b,r_bab,optimize=True)

               temp_2 = -np.einsum('ljzd,jzw->lwd',t2_1_ab,r_bab,optimize=True)

               s[s_a:f_a] -= 0.5*np.einsum('lwd,wlad->a',temp,v2e_vovv_a,optimize=True)
               s[s_a:f_a] -= 0.5*np.einsum('lwd,wlad->a',temp_1,v2e_vovv_ab,optimize=True)
               s[s_a:f_a] += 0.5*np.einsum('lwd,lwad->a',temp_2,v2e_ovvv_ab,optimize=True)

               if direct_adc.algorithm == "dynamical":
                   temp = np.zeros_like(r_aba,dtype=complex)
               else:
                   temp = np.zeros_like(r_aba)
               temp = -np.einsum('jlzd,jwz->lwd',t2_1_b,r_bbb_u,optimize=True)
               temp += -np.einsum('jlzd,jwz->lwd',t2_1_ab,r_aba,optimize=True)
               
               if direct_adc.algorithm == "dynamical":
                   temp_1 = np.zeros_like(r_bab,dtype=complex)
               else:
                   temp_1 = np.zeros_like(r_bab)
               temp_1 = -np.einsum('ljdz,jwz->lwd',t2_1_ab,r_bbb_u,optimize=True)
               temp_1 += -np.einsum('jlzd,jwz->lwd',t2_1_a,r_aba,optimize=True)

               temp_2 = -np.einsum('jldz,jzw->lwd',t2_1_ab,r_aba,optimize=True)

               s[s_b:f_b] -= 0.5*np.einsum('lwd,wlad->a',temp,v2e_vovv_b,optimize=True)
               #s[s_b:f_b] -= 0.5*np.einsum('lwd,lwda->a',temp_1,v2e_ovvv_ab,optimize=True)
               temp_a = temp_1.reshape(-1)
               temp_b = v2e_ovvv_ab.reshape(nocc_a*nvir_b*nvir_a,-1)
               s[s_b:f_b] -= 0.5*np.dot(temp_a,temp_b) 
               s[s_b:f_b] += 0.5*np.einsum('lwd,wlda->a',temp_2,v2e_vovv_ab,optimize=True)

################ ADC(3) ibc - a block ############################

               #t2_1_a_t = t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]]
               #temp = np.einsum('lmp,lmbi->bip',t2_1_a_t,v2e_oovo_a)
               #s[s_aaa:f_aaa] += 0.5*np.einsum('bip,b->ip',temp, r_a, optimize=True).reshape(-1)

               t2_1_a_t = t2_1_a[:,:,ab_ind_a[0],ab_ind_a[1]]
               temp = np.einsum('b,lmbi->lmi',r_a,v2e_oovo_a)
               s[s_aaa:f_aaa] += 0.5*np.einsum('lmi,lmp->ip',temp, t2_1_a_t, optimize=True).reshape(-1)

               #temp_1 = np.einsum('lmxy,lmbi->bixy',t2_1_ab,v2e_oovo_ab)
               #s[s_bab:f_bab] += np.einsum('bixy,b->ixy',temp_1, r_a, optimize=True).reshape(-1)
                  
               temp_1 = np.einsum('b,lmbi->lmi',r_a,v2e_oovo_ab)
               s[s_bab:f_bab] += np.einsum('lmi,lmxy->ixy',temp_1, t2_1_ab, optimize=True).reshape(-1)

               #t2_1_b_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]]
               #temp = np.einsum('lmp,lmbi->bip',t2_1_b_t,v2e_oovo_b)
               #s[s_bbb:f_bbb] += 0.5*np.einsum('bip,b->ip',temp, r_b, optimize=True).reshape(-1)

               t2_1_b_t = t2_1_b[:,:,ab_ind_b[0],ab_ind_b[1]]
               temp = np.einsum('b,lmbi->lmi',r_b,v2e_oovo_b)
               s[s_bbb:f_bbb] += 0.5*np.einsum('lmi,lmp->ip',temp, t2_1_b_t, optimize=True).reshape(-1)

               #temp_1 = np.einsum('mlyx,mlib->bixy',t2_1_ab,v2e_ooov_ab)
               #s[s_aba:f_aba] += np.einsum('bixy,b->ixy',temp_1, r_b, optimize=True).reshape(-1)

               temp_1 = np.einsum('b,mlib->mli',r_b,v2e_ooov_ab)
               s[s_aba:f_aba] += np.einsum('mli,mlyx->ixy',temp_1, t2_1_ab, optimize=True).reshape(-1)

               temp_1 = np.einsum('xlbd,b->lxd', v2e_vovv_a,r_a,optimize=True)
               temp_2 = np.einsum('xlbd,b->lxd', v2e_vovv_ab,r_a,optimize=True)
              
               temp  = np.einsum('lxd,ilyd->ixy',temp_1,t2_1_a,optimize=True)
               temp += np.einsum('lxd,ilyd->ixy',temp_2,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] += temp[:,ab_ind_a[0],ab_ind_a[1] ].reshape(-1)
               
               temp  = np.einsum('lxd,lidy->ixy',temp_1,t2_1_ab,optimize=True)
               temp  += np.einsum('lxd,ilyd->ixy',temp_2,t2_1_b,optimize=True)
               s[s_bab:f_bab] += temp.reshape(-1)

               temp_1 = np.einsum('xlbd,b->lxd', v2e_vovv_b,r_b,optimize=True)
               temp_2 = np.einsum('lxdb,b->lxd', v2e_ovvv_ab,r_b,optimize=True)
              
               temp  = np.einsum('lxd,ilyd->ixy',temp_1,t2_1_b,optimize=True)
               temp += np.einsum('lxd,lidy->ixy',temp_2,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] += temp[:,ab_ind_b[0],ab_ind_b[1] ].reshape(-1)
              
               temp  = np.einsum('lxd,ilyd->ixy',temp_1,t2_1_ab,optimize=True)
               temp  += np.einsum('lxd,ilyd->ixy',temp_2,t2_1_a,optimize=True)
               s[s_aba:f_aba] += temp.reshape(-1)

               temp_1 = np.einsum('ylbd,b->lyd', v2e_vovv_a,r_a,optimize=True)
               temp_2 = np.einsum('ylbd,b->lyd', v2e_vovv_ab,r_a,optimize=True)
              
               temp  = np.einsum('lyd,ilxd->ixy',temp_1,t2_1_a,optimize=True)
               temp += np.einsum('lyd,ilxd->ixy',temp_2,t2_1_ab,optimize=True)
               s[s_aaa:f_aaa] -= temp[:,ab_ind_a[0],ab_ind_a[1] ].reshape(-1)

               temp  = -np.einsum('lybd,b->lyd',v2e_ovvv_ab,r_a,optimize=True)
               temp_1= -np.einsum('lyd,lixd->ixy',temp,t2_1_ab,optimize=True)
               s[s_bab:f_bab] -= temp_1.reshape(-1)

               temp_1 = np.einsum('ylbd,b->lyd', v2e_vovv_b,r_b,optimize=True)
               temp_2 = np.einsum('lydb,b->lyd', v2e_ovvv_ab,r_b,optimize=True)
              
               temp  = np.einsum('lyd,ilxd->ixy',temp_1,t2_1_b,optimize=True)
               temp += np.einsum('lyd,lidx->ixy',temp_2,t2_1_ab,optimize=True)
               s[s_bbb:f_bbb] -= temp[:,ab_ind_b[0],ab_ind_b[1] ].reshape(-1)
              
               temp  = -np.einsum('yldb,b->lyd',v2e_vovv_ab,r_b,optimize=True)
               temp_1= -np.einsum('lyd,ildx->ixy',temp,t2_1_ab,optimize=True)
               s[s_aba:f_aba] -= temp_1.reshape(-1)

        if (direct_adc.algorithm == "dynamical"):
            s *= -1.0

        return s

    if (direct_adc.algorithm == "dynamical"):

        precond_ = -precond.copy()
        M_ab_a_ = -M_ab_a.copy()
        M_ab_b_ = -M_ab_b.copy()

    else :

        precond_ = precond.copy()
        M_ab_a_ = M_ab_a.copy()
        M_ab_b_ = M_ab_b.copy()

    return sigma_,precond_,(M_ab_a_,M_ab_b_)

#####################################################
# Solve linear equation using Conjugate Gradients #
#####################################################

def solve_conjugate_gradients(direct_adc,apply_H,precond,T,r,omega,orb):
    
    maxiter = direct_adc.maxiter

    iomega = omega + direct_adc.broadening*1j
     
    # Compute residual
    
    omega_H = iomega*r + apply_H(r)

    res = T - omega_H

    rms = np.linalg.norm(res)/np.sqrt(res.size)

    if rms < 1e-8:
        return r

    d = (precond+omega)**(-1)*res

    #d = res
    
    #delta_new = np.dot(np.ravel(res), np.ravel(res))
    delta_new = np.dot(np.ravel(res), np.ravel(d))

    conv = False

    for imacro in range(maxiter):

        Ad = iomega*d + apply_H(d)

        # q = A * d
        q = Ad

        # alpha = delta_new / d . q
        alpha = delta_new / (np.dot(np.ravel(d), np.ravel(q)))

        # x = x + alpha * d
        r = r + alpha * d
 
        res = res - alpha * q
        #s = (precond+1e-6)**(-1)*res
        s = (precond+omega)**(-1)*res

        #s = res

        delta_old = delta_new
        #delta_new = np.dot(np.ravel(res), np.ravel(res))
        delta_new = np.dot(np.ravel(res), np.ravel(s))

        beta = delta_new/delta_old

        #d = res + beta * d
        d = s + beta * d

        # Compute RMS of the residual
        rms = np.linalg.norm(res)/np.sqrt(res.size)
        
        print ("freq","   ",iomega,"   ","Iteration ", imacro, ": RMS = %8.4e" % rms)

        if abs(rms) < direct_adc.tol:
            conv = True
            break

    if conv:
        print ("Iterations converged")
    else:
        raise Exception("Iterations did not converge")

    return r

#############
# Davidson #
#############

def setup_davidson_ip(direct_adc, t_amp):

    apply_H = None
    precond = None

    apply_H, precond, M_ij = define_H_ip(direct_adc, t_amp)

#   x0 = compute_guess_vectors_M(direct_adc, precond, M_ij)
    x0 = compute_guess_vectors(direct_adc, precond)

    return apply_H, precond, x0

def setup_davidson_ea(direct_adc, t_amp):

    apply_H = None
    precond = None

    apply_H, precond, M_ab = define_H_ea(direct_adc, t_amp)

#    x0 = compute_guess_vectors_M(direct_adc, precond, M_ab)
    x0 = compute_guess_vectors(direct_adc, precond)

    return apply_H, precond, x0

###################
# Guess vectors 1 #
###################

def compute_guess_vectors(direct_adc, precond, ascending = True):

    sort_ind = None
    if ascending:
        sort_ind = np.argsort(precond)
    else:
        sort_ind = np.argsort(precond)[::-1]
    
    x0s = np.zeros((precond.shape[0], direct_adc.nstates))
    min_shape = min(precond.shape[0], direct_adc.nstates)
    x0s[:min_shape,:min_shape] = np.identity(min_shape)

    x0 = np.zeros((precond.shape[0], direct_adc.nstates))
    x0[sort_ind] = x0s.copy()

    x0s = []
    for p in range(x0.shape[1]):
        x0s.append(x0[:,p])

    return x0s

###################
# Guess vectors 2 #
###################

def compute_guess_vectors_M(direct_adc, precond, M, ascending = True):

    h0_dim_a = M[0].shape[0]
    h0_dim_b = M[1].shape[0]

    dim = precond.shape[0]

    evals_a, evecs_a = np.linalg.eigh(M[0])
    evals_b, evecs_b = np.linalg.eigh(M[1])

    precond_ = precond.copy()
    precond_[:h0_dim_a] = evals_a.copy()
    precond_[h0_dim_a:(h0_dim_a + h0_dim_b)] = evals_b.copy()

    sort_ind = None
    if ascending:
        sort_ind = np.argsort(precond_)
    else:
        sort_ind = np.argsort(precond_)[::-1]

    x0 = []
    for p in range(direct_adc.nstates):
        temp = np.zeros(dim)
        if sort_ind[p] < h0_dim_a:
            temp[:h0_dim_a] = evecs_a[:,sort_ind[p]].copy()
        elif h0_dim_a <= sort_ind[p] and sort_ind[p] < (h0_dim_a + h0_dim_b):
            temp[h0_dim_a:(h0_dim_a + h0_dim_b)] = evecs_b[:,sort_ind[p] - h0_dim_a].copy()
        else:
            temp[sort_ind[p]] = 1.0
        x0.append(temp)

    return x0

###################
# CVS Projector #
###################

def cvs_projector(direct_adc, r):

    ncore = direct_adc.n_core

    nocc_a = direct_adc.nocc_a
    nocc_b = direct_adc.nocc_b
    nvir_a = direct_adc.nvir_a
    nvir_b = direct_adc.nvir_b
    
    n_singles_a = nocc_a
    n_singles_b = nocc_b
    n_doubles_aaa = nocc_a * (nocc_a - 1) * nvir_a // 2
    n_doubles_bab = nvir_b * nocc_a * nocc_b
    n_doubles_aba = nvir_a * nocc_b * nocc_a
    n_doubles_bbb = nocc_b * (nocc_b - 1) * nvir_b // 2
    
    ij_a = np.tril_indices(nocc_a, k=-1)
    ij_b = np.tril_indices(nocc_b, k=-1)

    s_a = 0
    f_a = n_singles_a
    s_b = f_a
    f_b = s_b + n_singles_b
    s_aaa = f_b
    f_aaa = s_aaa + n_doubles_aaa
    s_bab = f_aaa
    f_bab = s_bab + n_doubles_bab
    s_aba = f_bab
    f_aba = s_aba + n_doubles_aba
    s_bbb = f_aba
    f_bbb = s_bbb + n_doubles_bbb

    Pr = r.copy()

    Pr[(s_a+ncore):f_a] = 0.0
    Pr[(s_b+ncore):f_b] = 0.0

    temp = np.zeros((nvir_a, nocc_a, nocc_a))
    temp[:,ij_a[0],ij_a[1]] = Pr[s_aaa:f_aaa].reshape(nvir_a,-1).copy()
    temp[:,ij_a[1],ij_a[0]] = -Pr[s_aaa:f_aaa].reshape(nvir_a,-1).copy()

    temp[:,ncore:,ncore:] = 0.0
    temp[:,:ncore,:ncore] = 0.0

    Pr[s_aaa:f_aaa] = temp[:,ij_a[0],ij_a[1]].reshape(-1).copy()

    temp = Pr[s_bab:f_bab].copy()
    temp = temp.reshape((nvir_b, nocc_a, nocc_b))
    temp[:,ncore:,ncore:] = 0.0
    temp[:,:ncore,:ncore] = 0.0

    Pr[s_bab:f_bab] = temp.reshape(-1).copy()

    temp = Pr[s_aba:f_aba].copy()
    temp = temp.reshape((nvir_a, nocc_b, nocc_a))
    temp[:,ncore:,ncore:] = 0.0
    temp[:,:ncore,:ncore] = 0.0

    Pr[s_aba:f_aba] = temp.reshape(-1).copy()

    temp = np.zeros((nvir_b, nocc_b, nocc_b))
    temp[:,ij_b[0],ij_b[1]] = Pr[s_bbb:f_bbb].reshape(nvir_b,-1).copy()
    temp[:,ij_b[1],ij_b[0]] = -Pr[s_bbb:f_bbb].reshape(nvir_b,-1).copy()

    temp[:,ncore:,ncore:] = 0.0
    temp[:,:ncore,:ncore] = 0.0

    Pr[s_bbb:f_bbb] = temp[:,ij_b[0],ij_b[1]].reshape(-1).copy()
    
    return Pr

def spec_factors_ip(direct_adc, t_amp, U):

    start_time = time.time()

    print ("\nComputing spectroscopic intensity:")

    nmo_a     = direct_adc.nmo_a
    nmo_b     = direct_adc.nmo_b
    nstates = direct_adc.nstates

    P = np.zeros((nstates))

    U = np.array(U)

    for orb in range(nmo_a):

            T_a = calculate_T_ip(direct_adc, t_amp, orb, spin = "alpha")

            T_a = np.dot(T_a, U.T)
            for i in range(nstates):
                P[i] += np.square(np.absolute(T_a[i]))

    for orb in range(nmo_b):

            T_b = calculate_T_ip(direct_adc, t_amp, orb, spin = "beta")

            T_b = np.dot(T_b, U.T)
            for i in range(nstates):
                P[i] += np.square(np.absolute(T_b[i]))

    print ("Time for Computing spectroscopic intensity:     %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    return P


def spec_factors_ea(direct_adc, t_amp, U):
    start_time = time.time()

    print ("\nComputing spectroscopic intensity:")

    nmo_a     = direct_adc.nmo_a
    nmo_b     = direct_adc.nmo_b
    nstates = direct_adc.nstates

    P = np.zeros((nstates))

    U = np.array(U)

    for orb in range(nmo_a):

            T_a = calculate_T_ea(direct_adc, t_amp, orb, spin = "alpha")

            T_a = np.dot(T_a, U.T)
            for i in range(nstates):
                P[i] += np.square(np.absolute(T_a[i]))

    for orb in range(nmo_b):

            T_b = calculate_T_ea(direct_adc, t_amp, orb, spin = "beta")

            T_b = np.dot(T_b, U.T)
            for i in range(nstates):
                P[i] += np.square(np.absolute(T_b[i]))

    print ("Time for Computing spectroscopic intensity:     %f sec\n" % (time.time() - start_time))
    sys.stdout.flush()

    return P

def filter_states(direct_adc, U_cvs, nroots):

    def pick_mom(w,v,nroots,local_var):
       
       # Compute overlap between v_cvs and v
       #for i in local_var:
       #    print (i)
       x_full = np.array(local_var['xs'])
       v_full_old = None
       vlast = local_var['vlast']
       if vlast is None:
           v_full_old = U_cvs.copy()
       else:
           print (vlast.shape)
           v_full_old = np.dot(vlast.T, x_full[:vlast.shape[0],:])
       v_full = np.dot(x_full.T,v)
       S = np.absolute(np.dot(v_full_old,v_full))
       P = S.sum(axis=0)
       idx = (-P).argsort()[:nroots]
#       U_cvs = v_full.T[idx,:].copy()
       #print (P)
       #print (P[idx])
       return w[idx], v[:,idx], idx
    return pick_mom
