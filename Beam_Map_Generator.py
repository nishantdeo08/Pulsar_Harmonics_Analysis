import numpy as np
import matplotlib.pyplot as plt 

# Load data
data = np.loadtxt("Extracted_RA_Dec_beam_index.txt", delimiter=",", skiprows=2, dtype=str)

RA = data[:, 0].astype(float)
DEC = data[:, 1].astype(float)
Index = data[:, 2]  # Keep as string for label

# Create plot
plt.figure(figsize=(12, 8))
plt.scatter(RA, DEC, s=40, color='darkblue')

# Add beam index labels
for i in range(len(RA)):
    plt.text(RA[i], DEC[i], Index[i], fontsize=8, ha='center')

# Plot styling
plt.xlabel("Right Ascension", fontsize=12)
plt.ylabel("Declination", fontsize=12)
plt.title("Beam Pattern plot", fontsize=14)
plt.grid(False)
plt.tight_layout()
# plt.gca().invert_xaxis()  # Optional: invert RA axis if you're matching astronomical convention

plt.show()
