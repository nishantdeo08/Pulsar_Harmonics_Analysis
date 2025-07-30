import glob
import numpy as np
import matplotlib.pyplot as plt

no_beam = 160
expected_DM = 73.758
DM_tol = 2.5
expected_period = 0.16678
Period_tol = 0.002

# def Harmonics_for_Beam(no_beam, expected_DM, DM_tol, expected_period, period_tol):

# Match all files like BM001_all_sifted_candidates.txt, etc.
matching_files = glob.glob('BM*_all_sifted_candidates.txt')

# Converting this list to a numpy array
matching_files = np.array(matching_files)

# Sort based on the number after 'BM'
matching_files = sorted(matching_files, key=lambda x: int(x.split('BM')[1].split('_')[0]))

# print(matching_files)

print("All the files are processed: ", len(matching_files) == no_beam)

fundamental_SNRs = []
harmonic_1_SNRs = []
harmonic_2_SNRs = []
harmonic_3_SNRs = []

for i in range(len(matching_files)):

#for i in range(10):
    data = np.loadtxt(matching_files[i], dtype=float, skiprows=1)

    if data.ndim == 1:
        data = data[np.newaxis, :]

    Period = data[:, 0]
    DM = data[:, 2]
    SNR = data[:, 3]

    condition = (DM > expected_DM - DM_tol) & (DM < expected_DM + DM_tol)
    idx_DM_match = np.where(condition)[0]
    harmonics = Period[idx_DM_match]

    #print(harmonics)

    # Fundamental
    condition_1 = (harmonics > expected_period - Period_tol) & (harmonics < expected_period + Period_tol)
    idx_fundamental = idx_DM_match[condition_1]
    fundamental_snr_values = SNR[idx_fundamental]
    if fundamental_snr_values.size > 0:
        max_snr = np.max(fundamental_snr_values)
        fundamental_SNRs.append(max_snr if max_snr >= 0 else np.nan)
    else:
        fundamental_SNRs.append(np.nan)

    #print(fundamental_snr_values)

    # Harmonic 1
    condition_2 = (harmonics > 0.5 * (expected_period - Period_tol)) & (harmonics < 0.5 * (expected_period + Period_tol))
    idx_harmonic_1 = idx_DM_match[condition_2]
    harmonic_1_snr_values = SNR[idx_harmonic_1]
    if harmonic_1_snr_values.size > 0:
        max_snr = np.max(harmonic_1_snr_values)
        harmonic_1_SNRs.append(max_snr if max_snr >= 0 else np.nan)
    else:
        harmonic_1_SNRs.append(np.nan)

    # Harmonic 2
    condition_3 = (harmonics > (1/3)*(expected_period - Period_tol)) & (harmonics < (1/3)*(expected_period + Period_tol))
    harmonic_2 = harmonics[condition_3]
    idx_harmonic_2 = idx_DM_match[condition_3]
    harmonic_2_snr_values = SNR[idx_harmonic_2]
    if harmonic_2_snr_values.size > 0:
        max_snr = np.max(harmonic_2_snr_values)
        harmonic_2_SNRs.append(max_snr if max_snr >= 0 else np.nan)
    else:
        harmonic_2_SNRs.append(np.nan)
 
    # Harmonic 3
    condition_4 = (harmonics > (1/4)*(expected_period - Period_tol)) & (harmonics < (1/4)*(expected_period + Period_tol))
    harmonic_3 = harmonics[condition_4]
    idx_harmonic_3 = idx_DM_match[condition_4]
    harmonic_3_snr_values = SNR[idx_harmonic_3]
    if harmonic_3_snr_values.size > 0:
        max_snr = np.max(harmonic_3_snr_values)
        harmonic_3_SNRs.append(max_snr if max_snr >= 0 else np.nan)
    else:
        harmonic_3_SNRs.append(np.nan)

# return fundamental_SNRs, harmonic_1_SNRs, harmonic_2_SNRs, harmonic_3_SNRs

# Load RA, DEC
data = np.loadtxt("Extracted_RA_Dec_beam_index.txt", delimiter=",", skiprows=2, dtype=str)
RA_ = data[:, 0].astype(float)
DEC_ = data[:, 1].astype(float)
Index = data[:, 2].astype(float)

# Get sort order by beam index
sorted_indices = np.argsort(Index)

# Apply sort order to RA, DEC, SNR arrays
RA = RA_[sorted_indices]
DEC = DEC_[sorted_indices]

# Truncate to number of processed beams
RA = RA[:len(fundamental_SNRs)]
DEC = DEC[:len(fundamental_SNRs)]

fundamental_SNRs = np.array(fundamental_SNRs)
harmonic_1_SNRs = np.array(harmonic_1_SNRs)
harmonic_2_SNRs = np.array(harmonic_2_SNRs)
harmonic_3_SNRs = np.array(harmonic_3_SNRs)

# Plot Fundamental SNR Heatmap
plt.figure(figsize=(10, 6))
sc1 = plt.scatter(RA, DEC, c=fundamental_SNRs, cmap='plasma', s=100)
plt.colorbar(sc1, label='SNR (Fundamental)')
plt.xlabel("RA")
plt.ylabel("DEC")
plt.title("SNR Map - Fundamental")
# plt.savefig("SNR_Map_Fundamental.png")
plt.show()

# Plot Harmonic 1 SNR Heatmap
plt.figure(figsize=(10, 6))
sc2 = plt.scatter(RA, DEC, c=harmonic_1_SNRs, cmap='plasma', s=100)
plt.colorbar(sc2, label='SNR (Harmonic 1)')
plt.xlabel("RA")
plt.ylabel("DEC")
plt.title("SNR Map - Harmonic 1")
# plt.savefig("SNR_Map_Harmonic1.png")
plt.show()

# Plot Harmonic 2 SNR Heatmap
plt.figure(figsize=(8, 6))
sc3 = plt.scatter(RA, DEC, c=harmonic_2_SNRs, cmap='plasma', s=100)
plt.colorbar(sc3, label='SNR (Harmonic 2)')
plt.xlabel("RA")
plt.ylabel("DEC")
plt.title("SNR Map - Harmonic 2")
# plt.savefig("SNR_Map_Harmonic1.png")
plt.show()

# Plot Harmonic 3 SNR Heatmap
plt.figure(figsize=(8, 6))
sc4 = plt.scatter(RA, DEC, c=harmonic_3_SNRs, cmap='plasma', s=100)
plt.colorbar(sc4, label='SNR (Harmonic 3)')
plt.xlabel("RA")
plt.ylabel("DEC")
plt.title("SNR Map - Harmonic 3")
# plt.savefig("SNR_Map_Harmonic1.png")
plt.show()

# print(fundamental_SNRs)