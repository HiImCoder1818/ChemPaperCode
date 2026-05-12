systems = {
    "He": "He 0 0 0",
    "Ne": "Ne 0 0 0",
    "Ar": "Ar 0 0 0",
    "Kr": "Kr 0 0 0",
    "Xe": "Xe 0 0 0",
    "Hg": "Hg 0 0 0",

    "C": "C 0 0 0",

    "O2": """
    O 0 0 0
    O 0 0 1.21
    """,

    "N2": """
    N 0 0 0
    N 0 0 1.10
    """,

    "Cl2": """
    Cl 0 0 0
    Cl 0 0 1.99
    """,

    "Br2": """
    Br 0 0 0
    Br 0 0 2.28
    """,

    "I2": """
    I 0 0 0
    I 0 0 2.67
    """
}
import numpy as np
from pyscf import gto, scf, tdscf

# -------------------------------
# 1. RUN HF + TDHF
# -------------------------------
#"cc-pvdz"
#"aug-cc-pvdz"
#"lanl2dz"
basis = "aug-cc-pvdz"
ecp = "aug-cc-pvdz"

def compute_excited_states(atom, basis=basis, nstates=30):

    mol = gto.M(
        atom=atom,
        basis=basis,
        ecp=ecp
    )
    mol.charge = +1
    mol.spin = 1

    mf = scf.RHF(mol)
    mf.kernel()

    td = tdscf.TDHF(mf)
    td.nstates = nstates

    exc_energies, _ = td.kernel()

    # convert Hartree → eV
    exc_energies = np.array(exc_energies) * 27.2114

    return exc_energies


# -------------------------------
# 2. BUILD DISCHARGE SPECTRUM
# -------------------------------
def discharge_spectrum(exc_energies, temp=1.0):

    E = np.array(exc_energies)

    # pseudo population distribution (proxy for plasma)
    population = np.exp(-E / temp)
    population /= np.sum(population)

    lines = []

    n = len(E)

    for i in range(n):
        for j in range(n):

            if E[j] > E[i]:

                dE = E[j] - E[i]
                wavelength = 1240.0 / dE  # nm

                # visible window
                if 300 <= wavelength <= 800:

                    intensity = population[i]

                    lines.append((wavelength, intensity))

    # sort by wavelength
    lines.sort(key=lambda x: x[0])

    return lines


# -------------------------------
# 3. FULL PIPELINE WRAPPER
# -------------------------------
def run_spectrum(atom, basis=basis, nstates=30):

    exc = compute_excited_states(atom, basis, nstates)

    lines = discharge_spectrum(exc)

    return lines


# -------------------------------
# 4. EXAMPLE USAGE
# -------------------------------
if __name__ == "__main__":
    lines = run_spectrum(systems["Ne"])

    print("Visible emission lines (nm, intensity):\n")

    for lam, inten in lines[:30]:
        print(f"{lam:.1f} nm   {inten:.4f}")