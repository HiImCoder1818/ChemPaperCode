from pyscf import gto, dft, tddft
import numpy as np

# -------------------------------
# SYSTEM DATABASE
# -------------------------------
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

# -------------------------------
# PARAMETERS
# -------------------------------
#lanl2dz
#aug-cc-pVDZ
BASIS = "lanl2dz"
ECP = "lanl2dz"
FUNCTIONAL = "CAM-B3LYP"

# -------------------------------
# 1. DFT + TDDFT
# -------------------------------
def compute_excited_states(atom, basis=BASIS, functional=FUNCTIONAL, nstates=50):

    mol = gto.M(
        atom=atom,
        basis=basis,
        ecp=ECP
    )

    ks = dft.RKS(mol)
    ks.xc = functional
    ks.conv_tol = 1e-8
    ks.max_cycle = 200
    ks.kernel()

    td = tddft.TDDFT(ks)
    td.nstates = nstates
    td.kernel()

    energies = np.array(td.e) * 27.2114  # Hartree → eV

    osc = np.array(td.oscillator_strength())

    return energies, osc


# -------------------------------
# 2. PLASMA-LIKE POPULATION MODEL
# -------------------------------
def plasma_spectrum(E, osc, temp=2.0):

    # Boltzmann-like weighting (your idea, stabilized)
    pop = np.exp(-E / temp)
    pop = pop / np.sum(pop)

    lines = []

    n = len(E)

    for i in range(n):
        for j in range(n):

            if E[j] > E[i]:

                dE = E[j] - E[i]
                wavelength = 1240.0 / dE

                # visible window
                if 200 <= wavelength <= 1000:

                    intensity = pop[i] * osc[i] * osc[j]

                    lines.append((wavelength, intensity))

    return sorted(lines, key=lambda x: x[0])


# -------------------------------
# 3. FULL PIPELINE
# -------------------------------
def run(atom):

    E, osc = compute_excited_states(atom)

    lines = plasma_spectrum(E, osc)

    return lines


# -------------------------------
# 4. EXECUTION
# -------------------------------
if __name__ == "__main__":

    lines = run(systems["Hg"])

    print("Visible emission lines (nm, intensity):\n")

    for lam, inten in lines[:30]:
        print(f"{lam:.1f} nm   {inten:.6f}")