import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm, genlaguerre, factorial
from mpl_toolkits.mplot3d import Axes3D

# ============================================================
# Hydrogen Orbital Monte Carlo Visualization
# ------------------------------------------------------------
# Change n, l, m below to visualize different orbitals
#
# Examples:
# 1s  -> n=1, l=0, m=0
# 2s  -> n=2, l=0, m=0
# 2pz -> n=2, l=1, m=0
# 2px -> n=2, l=1, m=1
# 3dz2 -> n=3, l=2, m=0
# ============================================================

n = 2
l = 1
m = 0

# Bohr radius (scaled)
a0 = 2.0

# Number of accepted points
N_POINTS = 40000

# Sampling cube size
BOUND = 20

# ============================================================
# Hydrogen radial function R_nl(r)
# ============================================================

def R_nl(n, l, r):

    rho = 2 * r / (n * a0)

    prefactor = np.sqrt(
        (2 / (n * a0))**3 *
        factorial(n - l - 1) /
        (2 * n * factorial(n + l))
    )

    laguerre = genlaguerre(n - l - 1, 2*l + 1)(rho)

    return (
        prefactor *
        np.exp(-rho / 2) *
        rho**l *
        laguerre
    )

# ============================================================
# Full hydrogen wavefunction ψ_nlm
# ============================================================

def psi_nlm(n, l, m, x, y, z):

    r = np.sqrt(x**2 + y**2 + z**2)

    # avoid divide-by-zero issues
    theta = np.arccos(np.divide(
        z,
        r,
        out=np.zeros_like(r),
        where=r != 0
    ))

    phi = np.arctan2(y, x)

    radial = R_nl(n, l, r)

    angular = sph_harm(m, l, phi, theta)

    return radial * angular

# ============================================================
# Rejection sampling from |ψ|²
# ============================================================

accepted_x = []
accepted_y = []
accepted_z = []
accepted_density = []

batch = 200000

while len(accepted_x) < N_POINTS:

    # Random candidate points
    x = np.random.uniform(-BOUND, BOUND, batch)
    y = np.random.uniform(-BOUND, BOUND, batch)
    z = np.random.uniform(-BOUND, BOUND, batch)

    psi = psi_nlm(n, l, m, x, y, z)

    density = np.abs(psi)**2

    # Normalize probabilities
    density /= np.max(density)

    # Random accept/reject
    keep = np.random.rand(batch) < density

    accepted_x.extend(x[keep])
    accepted_y.extend(y[keep])
    accepted_z.extend(z[keep])
    accepted_density.extend(density[keep])

# Trim exact count
accepted_x = np.array(accepted_x[:N_POINTS])
accepted_y = np.array(accepted_y[:N_POINTS])
accepted_z = np.array(accepted_z[:N_POINTS])
accepted_density = np.array(accepted_density[:N_POINTS])

# ============================================================
# Plot
# ============================================================

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(111, projection='3d')

sc = ax.scatter(
    accepted_x,
    accepted_y,
    accepted_z,
    c=accepted_density,
    cmap='coolwarm',
    s=1,
    alpha=0.15
)

ax.set_title(f"Hydrogen Orbital n={n}, l={l}, m={m}")

ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")

# Clean visual appearance
ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.view_init(elev=20, azim=40)

plt.colorbar(sc, label=r"$|\psi|^2$")

plt.tight_layout()
plt.show()