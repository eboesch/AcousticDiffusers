import matplotlib.pyplot as plt
import numpy as np

# ua skyline (https://ua-acoustics.com/acoustic-diffuser-skyline)
ua = [0, 0, 0.38, 0.46, 0.49, 0.53, 0.59, 0.61, 0.62, 0.64, 0.63, 0.65, 0.69, 0.62, 0.6, 0.57, 0.54]
# bluetone skyline (https://www.btacoustics.com/skyline)
bt = [0, 0, 0, 0, 0.19, 0.21, 0.24, 0.25, 0.29, 0.39, 0.51, 0.66, 0.79, 0.81, 0.82, 0.83, 0.81]
# rpg skyline (https://www.rpgacoustic.com/wp-content/uploads/2022/06/Skyline-T894-Data-Sheet.pdf)
rpg = [0.14, 0.11, 0.03, 0, 0.01, 0.01, 0.02, 0.03, 0.06, 0.15, 0.32, 0.43, 0.51, 0.57, 0.65, 0.5, 0.41]
# 1d qrd (aaad)
qrd = [0.07, 0.01, 0.02, 0, 0.01, 0.01, 0.01, 0.07, 0.16, 0.21, 0.12, 0.1, 0.07, 0.23, 0.39, 0.04, 0.19]
# 1d prd (aaad)
prd = [0.07, 0.01, 0.02, 0, 0.01, 0.02, 0.04, 0.03, 0.06, 0.33, 0.13, 0.14, 0.2, 0.27, 0.1, 0.35, 0.21]

plot_egs = False

lss = ['-', '--', ':', '-.']

# plt.style.use('dark_background')

freqs = [100, 125, 160, 200, 250, 315, 400, 500, 630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000]

# to_plot = [str(i) for i in range(0, 100)] + ["prd_l", "ptlight_cat"]

ref = None
# ref = "rect_0.6"
title = None

# lbls = {tp: tp for tp in to_plot}

dat = []
with open('coeffs.csv') as f:
    dat = f.readlines()
dat = [line.split(",") for line in dat]
coef_dat = {line[0]: np.array(list(map(float, line[1:]))) for line in dat}

to_plot = list(coef_dat.keys())
lbls = {tp: tp for tp in to_plot}

if ref is not None:
    for k in coef_dat:
        if k != ref:
            if len(coef_dat[k]) == len(coef_dat[ref]):
                coef_dat[k] = np.maximum(0, (coef_dat[k] - coef_dat[ref]) / (1 - coef_dat[ref]))

plt.ioff()
plt.figure(dpi=300)
plt.ylim(0, 1)
plt.yticks(np.arange(0, 1.1, 0.1))
for lbl in to_plot:
    if ref is None or lbl != ref:
        avg = np.mean(coef_dat[lbl])
        plt.plot(freqs, coef_dat[lbl], label=f"{lbls[lbl]} - {avg:.3f}")

# plt.plot(sixth_freqs, coef_dat["ac_0.02_sixth"], label="ac_0.02_sixth")
if plot_egs:
    plt.plot(freqs, ua, label="ua skyline")
    plt.plot(freqs, bt, label="bluetone skyline")
    plt.plot(freqs, rpg, label="rpg skyline")
    plt.plot(freqs, qrd, label="qrd")
    plt.plot(freqs, prd, label="prd")

if title:
    plt.title(title)

# plt.legend(prop={"size": 6})
plt.grid(True, which="both", ls=":")
plt.savefig(f"out.png")
# plt.show()
