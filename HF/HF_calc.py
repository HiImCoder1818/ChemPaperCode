"""
Hartree-Fock allowed transitions for a single atom using PySCF.

Pipeline:
  1. Run HF -> get occupied and virtual eigenpairs
  2. Compute dipole matrix D_uv = <chi_u | r | chi_v> analytically
  3. For each occupied -> virtual pair compute mu_ia = C_i^T D C_a
  4. If |mu_ia|^2 > threshold -> allowed, compute delta_E and wavelength

Install: pip install pyscf
"""

import numpy as np

ELEMENT   = "Ar"
THRESHOLD = 1e-6  # |mu_ia|^2 cutoff for allowed vs forbidden

def get_orbital_name(ao_labels, C, mo_idx):
    dominant = np.argmax(np.abs(C[:, mo_idx]))
    label = ao_labels[dominant].strip()
    return " ".join(label.split()[2:])  # e.g. "1s", "2px"

def run_hf(element):
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.atom = f"{element} 0 0 0"
    mol.basis = "aug-cc-pVDZ"
    mol.charge = 0
    mol.spin = 0
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    return mol, mf

def compute_transitions(mol, mf):
    energies  = mf.mo_energy   # orbital energies (eigenvalues)
    occ       = mf.mo_occ      # occupation numbers
    C         = mf.mo_coeff    # coefficient matrix
    ao_labels = mol.ao_labels()

    # Split into occupied and virtual indices
    occ_idx  = [i for i, o in enumerate(occ) if o > 0]
    virt_idx = [i for i, o in enumerate(occ) if o == 0]

    # --- Compute dipole matrix in AO basis ---
    # D_uv = <chi_u | x | chi_v>,  <chi_u | y | chi_v>,  <chi_u | z | chi_v>
    # PySCF computes these analytically from the Gaussian integrals
    with mol.with_common_orig([0, 0, 0]):
        D_ao = mol.intor("int1e_r", comp=3)  # shape: (3, n_ao, n_ao)

    # Transform dipole matrix from AO basis to MO basis
    # D_MO = C^T @ D_AO @ C  for each component x,y,z
    D_mo = np.array([C.T @ D_ao[k] @ C for k in range(3)])  # shape: (3, n_mo, n_mo)

    # --- Evaluate each occupied -> virtual pair ---
    transitions = []

    for i in occ_idx:
        for a in virt_idx:
            # mu_ia is a 3-vector (x, y, z components)
            mu_ia = D_mo[:, i, a]          # shape: (3,)

            # transition probability
            mu_sq = np.dot(mu_ia, mu_ia)   # |mu_ia|^2, scalar

            if mu_sq > THRESHOLD:
                delta_E_Ha = energies[a] - energies[i]   # transition energy in Hartree
                delta_E_eV = delta_E_Ha * 27.2114
                wavelength = 1240 / delta_E_eV            # lambda = hc/E in nm

                name_i = get_orbital_name(ao_labels, C, i)
                name_a = get_orbital_name(ao_labels, C, a)

                transitions.append({
                    "from":       name_i,
                    "to":         name_a,
                    "dE_eV":      delta_E_eV,
                    "dE_Ha":      delta_E_Ha,
                    "lambda_nm":  wavelength,
                    "mu_sq":      mu_sq,
                })

    # Sort by transition energy
    transitions.sort(key=lambda t: t["dE_eV"])
    return transitions

def print_results(element, transitions):
    print(f"\n{'='*75}")
    print(f"  Allowed HF transitions for {element}")
    print(f"{'='*75}")
    print(f"  {'from':>6}  {'to':>6}  {'ΔE (eV)':>10}  {'λ (nm)':>10}  {'|μ|²':>12}")
    print(f"  {'-'*6}  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*12}")

    for t in transitions:
        print(f"  {t['from']:>6}  {t['to']:>6}  "
              f"{t['dE_eV']:>10.4f}  {t['lambda_nm']:>10.2f}  {t['mu_sq']:>12.6f}")

    print(f"{'='*75}")
    print(f"  Total allowed transitions: {len(transitions)}")
    print(f"{'='*75}\n")

if __name__ == "__main__":
    mol, mf = run_hf(ELEMENT)
    transitions = compute_transitions(mol, mf)
    print_results(ELEMENT, transitions)