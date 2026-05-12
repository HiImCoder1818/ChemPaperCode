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
sigma = 2

# --- element peak definitions ---
spectra = {
    "Neon": [
        (305, 1),
        (395, 1),
        (412, 1),
    ],

    "Argon": [
        (305, 1),
        (420, 1),
        (711, 1),
    ],

    "Mercury": [
        (513, 1),
        (560, 1),
    ]
}

# --- plotting ---
fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)

for ax, (name, peaks) in zip(axes, spectra.items()):
    intensity = np.zeros_like(wavelength)

    for center, amp in peaks:
        intensity += amp * np.exp(-((wavelength - center) ** 2) / (2 * sigma ** 2))

    for i in range(len(wavelength) - 1):
        rgb = wavelength_to_rgb(wavelength[i])
        ax.plot(
            wavelength[i:i+2],
            intensity[i:i+2],
            color=rgb,
            linewidth=2
        )
        ax.fill_between(
            wavelength[i:i+2],
            intensity[i:i+2],
            0,
            color=rgb,
            alpha=0.3
        )

    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Intensity")
    ax.set_title(f"{name} Emission Spectrum")

axes[-1].set_xlabel("Wavelength (nm)")

plt.tight_layout()
plt.savefig("calculated_emission_spectra.pdf")
plt.show()