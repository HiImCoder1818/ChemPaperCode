import numpy as np

ELEMENT = "Ar"
N_STATES = 30
THRESHOLD = 1e-6

def get_orbital_name(ao_labels, C, mo_idx):
    dominant = np.argmax(np.abs(C[:, mo_idx]))
    label = ao_labels[dominant].strip()
    return " ".join(label.split()[2:])

def run_hf(element):
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.atom = f"{element} 0 0 0"

    # Better excited-state basis
    mol.basis = "aug-cc-pVTZ"

    mol.charge = 0
    mol.spin = 0
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf.kernel()

    return mol, mf

def run_tdhf(mf, nstates):
    from pyscf import tdscf

    td = tdscf.TDHF(mf)
    td.nstates = nstates
    td.kernel()

    return td

def compute_transition_dipoles(mol, mf, td):

    C = mf.mo_coeff
    occ = mf.mo_occ
    ao_labels = mol.ao_labels()

    occ_idx  = [i for i,o in enumerate(occ) if o > 0]
    virt_idx = [i for i,o in enumerate(occ) if o == 0]

    # AO dipole integrals
    with mol.with_common_orig([0,0,0]):
        D_ao = mol.intor("int1e_r", comp=3)

    # Transform AO -> MO
    D_mo = np.array([
        C.T @ D_ao[k] @ C
        for k in range(3)
    ])

    results = []

    # TDHF amplitudes
    # td.xy[state] = (X, Y)
    for state_idx, ((X, Y), excitation_energy) in enumerate(zip(td.xy, td.e)):

        mu = np.zeros(3)

        # X/Y matrices shape:
        # (nocc, nvirt)
        for i_occ, i in enumerate(occ_idx):
            for a_virt, a in enumerate(virt_idx):

                coeff = X[i_occ, a_virt] + Y[i_occ, a_virt]

                mu += coeff * D_mo[:, i, a]

        mu_sq = np.dot(mu, mu)

        if mu_sq > THRESHOLD:

            energy_ev = excitation_energy * 27.2114
            wavelength = 1240 / energy_ev

            results.append({
                "state": state_idx + 1,
                "energy_ev": energy_ev,
                "wavelength_nm": wavelength,
                "mu_sq": mu_sq,
                "mu_vector": mu,
            })

    return results

def print_results(results):

    print("\n" + "="*90)
    print("TDHF Transition Spectrum")
    print("="*90)

    print(
        f"{'State':>6}  "
        f"{'Energy (eV)':>14}  "
        f"{'λ (nm)':>12}  "
        f"{'|μ|²':>12}"
    )

    print("-"*90)

    for r in results:

        print(
            f"{r['state']:>6}  "
            f"{r['energy_ev']:>14.4f}  "
            f"{r['wavelength_nm']:>12.2f}  "
            f"{r['mu_sq']:>12.6f}"
        )

    print("="*90)

if __name__ == "__main__":

    mol, mf = run_hf(ELEMENT)

    print("Running TDHF...")
    td = run_tdhf(mf, N_STATES)

    results = compute_transition_dipoles(mol, mf, td)

    print_results(results)