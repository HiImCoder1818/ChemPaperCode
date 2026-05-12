import numpy as np
import matplotlib.pyplot as plt

# --- wavelength to RGB mapping ---
def wavelength_to_rgb(wavelength):
    if 380 <= wavelength <= 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= wavelength <= 490:
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength <= 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength <= 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= wavelength <= 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif 645 <= wavelength <= 700:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = G = B = 0.0

    return (R, G, B)

# --- wavelength axis ---
wavelength = np.linspace(300, 750, 2000)

# --- approximate spectral peaks (edit these) ---
"""
Tube A:
peaks = [
    (500, 0.3),
    (550, 0.5),
    (600, 0.75),
    (675, 0.4),
]

Tube B:
peaks = [
    (500, 0.4),
    (550, 0.8),
    (610, 0.5),
]

Tube C:
peaks = [
    (400, 0.1),
    (425, 0.1),
    (450, 0.1),
    (475, 0.1),
    (550, 0.05),
    (590, 0.2),
    (650, 0.2),
]

A: Neon
B: Merucy
C: Argon

Calcuations:
Neon:
peaks = [
    (305, 1),
    (395, 1),
    (412, 1),
]

Argon:
peaks = [
    (305, 1),
    (420, 1),
    (711, 1),
]

Mercury:
peaks = [
    (513, 1),
    (560, 1),
]
"""
peaks = [
    (513, 1),
    (560, 1),
]

sigma = 2  # broadening (instrument + eye resolution)

# --- build spectrum ---
intensity = np.zeros_like(wavelength)

for center, amp in peaks:
    intensity += amp * np.exp(-((wavelength - center) ** 2) / (2 * sigma ** 2))

# --- plot colored spectrum ---
plt.figure(figsize=(10, 3))

for i in range(len(wavelength) - 1):
    rgb = wavelength_to_rgb(wavelength[i])
    plt.plot(
        wavelength[i:i+2],
        intensity[i:i+2],
        color=rgb,
        linewidth=2
    )
    plt.fill_between(
        wavelength[i:i+2],
        intensity[i:i+2],
        0,
        color=rgb,
        alpha=0.3
    )

plt.ylim(0, 1)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Intensity (a.u.)")
plt.title("Emission Spectrum in Tube A")

# plt.savefig("tubeA_spectrum.pdf", dpi=300, bbox_inches="tight")

plt.show()