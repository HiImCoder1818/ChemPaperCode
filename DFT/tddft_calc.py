import numpy as np
from pyscf import gto, dft, tddft

#"aug-cc-pvdz"
#"lanl2dz"
ELEMENT = "Hg"
BASIS = "lanl2dz"
ECP = "lanl2dz"
FUNCTIONAL = "CAM-B3LYP"
THRESHOLD = 1e-3

# def get_X(state_idx):
#         X_a, X_b = td.xy[state_idx]
#         return X_a  # shape (nocc, nvir)

# def excited_tdm(i, j):
#     Xi = get_X(i)  # (nocc, nvir)
#     Xj = get_X(j)  # (nocc, nvir)
#     mu = np.zeros(3)
#     for k in range(3):
#         # D_mo virtual-virtual block and occupied-occupied block
#         D_vv = D_mo[k, nocc:, nocc:]  # (nvir, nvir)
#         D_oo = D_mo[k, :nocc, :nocc]  # (nocc, nocc)
#         # one electron transition: virtual-virtual contribution
#         mu[k] += np.einsum('ia,ab,ib->', Xi, D_vv, Xj)
#         # occupied-occupied contribution
#         mu[k] -= np.einsum('ia,ij,ja->', Xi, D_oo, Xj)
#     return mu

def run_dft_tddft(element, basis, functional, charge=0, spin=0, ecp=None):

    # --------------------
    # 1. Build molecule
    # --------------------
    mol = gto.Mole()
    mol.atom = f"{element} 0 0 0"
    mol.basis = basis
    mol.ecp = ecp
    mol.charge = charge
    mol.spin = spin
    mol.symmetry = False
    mol.verbose = 0
    mol.build()

    # --------------------
    # 2. DFT
    # --------------------
    ks = dft.UKS(mol)
    ks.xc = functional
    ks.conv_tol = 1e-9
    ks.max_cycle = 200
    ks.kernel()

    print(f"DFT Energy: {ks.e_tot:.6f} Ha")

    # --------------------
    # 3. TDDFT (on KS)
    # --------------------
    td = tddft.TDA(ks)
    td.nstates = 30
    td.kernel()

    exc_energies = np.array(td.e) * 27.2114
    osc = td.oscillator_strength()
    wavelengths = []

    print("\nAllowed transitions:")
    print("State   Wavelength(nm)     Energy (eV)      osc")
    
    for i in range(len(exc_energies)):
        for j in range(len(exc_energies)):
            E_i = exc_energies[i]
            E_j = exc_energies[j]
            osc_ij = (osc[i] + osc[j]) / 2

            if E_i > E_j and osc_ij > THRESHOLD:
                dE = E_i - E_j
                wavelength = 1240 / dE
                
                if 300 <= wavelength <= 800:
                    wavelengths.append(float(wavelength))
                    print(f"{i}->{j}    {wavelength}      {dE:10.4f}     {osc_ij:10.4f}")

    print("State   Wavelength(nm)     Energy (eV)   Oscillator strength")
    for i in range(len(exc_energies)):
        wavelength = 1240 / exc_energies[i]
        print(f"{i:3d}    {wavelength}      {exc_energies[i]:10.4f}     {osc[i]:.5f}")

    return wavelengths


if __name__ == "__main__":
    w = run_dft_tddft(ELEMENT, BASIS, FUNCTIONAL, charge=+3, spin=3, ecp=ECP)
    print(w)
    