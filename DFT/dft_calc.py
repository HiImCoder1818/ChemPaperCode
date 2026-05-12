"""
DFT ground state energy for Argon using PySCF.

Pipeline:
  1. Define atom + basis set (Gaussians)
  2. Run KS-DFT SCF loop
  3. Print ground state energy and converged orbitals

Install: pip install pyscf
"""

import numpy as np
from pyscf import gto, dft

ELEMENT  = "Ar"
BASIS    = "def2-svp"    # double-zeta basis, better than STO-3G
FUNCTIONAL = "b3lyp"     # hybrid functional, standard for most systems

def run_dft(element, basis, functional):

    # --- Step 1: Build the molecule ---
    # Single atom at origin, no geometry needed
    mol = gto.Mole()
    mol.atom    = f"{element} 0 0 0"
    mol.basis   = basis        # Gaussian basis set {chi_mu}
    mol.charge  = 0
    mol.spin    = 0            # closed shell
    mol.verbose = 3            # print SCF iterations so we can see the loop
    mol.build()

    print(f"\n{'='*60}")
    print(f"  DFT ground state calculation")
    print(f"  Element    : {element}")
    print(f"  Basis      : {basis}")
    print(f"  Functional : {functional}")
    print(f"  Basis fns  : {mol.nao} Gaussian basis functions")
    print(f"{'='*60}\n")

    # --- Step 2: Set up KS-DFT ---
    ks = dft.RKS(mol)          # Restricted KS (closed shell)
    ks.xc = functional         # choose Exc functional

    # This runs the SCF loop:
    #   guess C
    #   -> build rho from orbitals
    #   -> build F^KS from rho  (T, Vne, J, Vxc)
    #   -> solve F^KS C = SC eps  -> new C
    #   -> repeat until rho converges
    ks.kernel()

    # --- Step 3: Extract results ---
    E0       = ks.e_tot        # ground state energy E[rho] in Hartree
    energies = ks.mo_energy    # KS orbital energies epsilon_i
    occ      = ks.mo_occ       # occupation numbers
    C        = ks.mo_coeff     # converged C matrix (columns = orbitals)

    # Get AO labels to name each orbital
    ao_labels = mol.ao_labels()

    # Split occupied vs virtual
    occ_idx  = [i for i, o in enumerate(occ) if o > 0]
    virt_idx = [i for i, o in enumerate(occ) if o == 0]

    # --- Print occupied eigenpairs ---
    print(f"\n{'='*60}")
    print(f"  Converged KS occupied eigenpairs")
    print(f"{'='*60}")
    print(f"  {'#':>3}  {'orbital':>8}  {'ε (Ha)':>12}  {'ε (eV)':>10}")
    print(f"  {'-'*3}  {'-'*8}  {'-'*12}  {'-'*10}")

    for idx, i in enumerate(occ_idx):
        dominant     = np.argmax(np.abs(C[:, i]))
        label        = ao_labels[dominant].strip()
        orbital_name = " ".join(label.split()[2:])
        e_Ha         = energies[i]
        e_eV         = e_Ha * 27.2114
        print(f"  {idx+1:>3}  {orbital_name:>8}  {e_Ha:>12.6f}  {e_eV:>10.4f}")

    # --- Print ground state energy breakdown ---
    print(f"\n{'='*60}")
    print(f"  Ground state energy breakdown")
    print(f"{'='*60}")
    print(f"  E[rho] total      : {E0:.6f} Ha  ({E0*27.2114:.4f} eV)")
    print(f"  HOMO energy       : {energies[occ_idx[-1]]:.6f} Ha  "
          f"({energies[occ_idx[-1]]*27.2114:.4f} eV)")
    print(f"  LUMO energy       : {energies[virt_idx[0]]:.6f} Ha  "
          f"({energies[virt_idx[0]]*27.2114:.4f} eV)")
    print(f"  HOMO-LUMO gap     : "
          f"{(energies[virt_idx[0]]-energies[occ_idx[-1]])*27.2114:.4f} eV")
    print(f"  SCF converged     : {ks.converged}")
    print(f"{'='*60}\n")

    return E0, energies, C

if __name__ == "__main__":
    E0, energies, C = run_dft(ELEMENT, BASIS, FUNCTIONAL)

"""
1s	-113.80 Ha	-3096 eV
2s	-10.79 Ha	-294 eV
2p	-8.44 Ha	-230 eV
3s	-0.883 Ha	-24.0 eV

 1s   -115.036164  -3130.2951
 2s    -11.113622   -302.4172
2p     -8.671077   -235.9521
 3s     -0.944018    -25.6881
"""

"""
from pyscf import tddft

# run DFT first — same as before
ks = dft.RKS(mol)
ks.xc = "b3lyp"
ks.kernel()

# run TDDFT on top
td = tddft.TDDFT(ks)
td.nstates = 10        # how many excited states you want
td.kernel()

# td.e       -> transition energies in Hartree
# td.oscillator_strength() -> oscillator strengths (intensity)
"""