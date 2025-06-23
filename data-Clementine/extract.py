import numpy as np
import matplotlib.pyplot as plt

fname = "raw_data.txt"

# Extraction
base_data = np.loadtxt(fname, skiprows=1,
                       usecols=[1, 2, 3, 4])  # I, t, d, P
# Remove rows with nan values
complete_data = []
for i, l in enumerate(base_data):
    if not (np.isnan(l).any()):
        complete_data.append(l)
# Sort wrt current density
complete_data.sort(key=lambda x:x[0])
unique_I = [complete_data[0][0]]
for l in complete_data:
    if l[0] != unique_I[-1]:
        unique_I.append(l[0])
# Separate values for
# I: current [mA/cm^2]
# t: time [s]
# d: final thickness [nm]
# P: porosity [%]
# r: etch_rate [nm/s]
I, t, d, P = np.transpose(complete_data)
r = d/t
# Define min, mean and max values
P1 = []
P2 = []
P3 = []
r1 = []
r2 = []
r3 = []
current_I = I[0]
current_I_index = 0
for I_test in unique_I:
    keep_ind = (I == I_test)
    P_test = P[keep_ind]
    r_test = r[keep_ind]
    P1.append(min(P_test))
    r1.append(min(r_test))
    P3.append(max(P_test))
    r3.append(max(r_test))
    P2.append(sum(P_test)/len(P_test))
    r2.append(sum(r_test)/len(r_test))

# Show this in a graph
fig, ax = plt.subplots(2, 1, sharex=True,
                       figsize=[6.4, 6.4])  # default: [6.4, 4.8]
ax[0].set_ylabel("porosity [%]")
ax[1].set_ylabel("etch rate [nm/s]")
ax[1].set_ylabel("current density [mA/cm^2]")

ax[0].plot(unique_I, P2, "k.-")
ax[0].plot(unique_I, P1, "k:")
ax[0].plot(unique_I, P3, "k:")
ax[1].plot(unique_I, r2, "k.-")
ax[1].plot(unique_I, r1, "k:")
ax[1].plot(unique_I, r3, "k:")

plt.savefig("plots_P(J)_r(J).jpg")
plt.show()

# Save mean values in text files
nm_per_s = 1e9
mA_per_cm2 = 1e-1
percent = 100.
# in folder "data-Clementine":
unique_I = np.array(unique_I)
P2 = np.array(P2)
r2 = np.array(r2)
np.savetxt("j-rho_data.txt", np.transpose([P2/percent, unique_I/mA_per_cm2]), header="rho[-] J[A/m^2]")
np.savetxt("j-etch_data.txt", np.transpose([r2/nm_per_s, unique_I/mA_per_cm2]), header="etch_rate[m/s] J[A/m^2]")




