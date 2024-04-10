import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import FormatStrFormatter
import pandas as pd

fig = plt.figure(figsize=(10, 7.5), facecolor='w', edgecolor='k')
fig.subplots_adjust(left=0.04, bottom=0.08, right=0.98, top=0.98, wspace=0.25, hspace=0.32)

ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

titlefont = 16
tickfont = 14
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
ax1.set_title('(a)  with ESD noise', fontsize=titlefont)
ax2.set_title('(b)  with ITM error', fontsize=titlefont)
ax3.set_title('(c)  Relative acc with ESD noise', fontsize=titlefont)
ax4.set_title('(d)  Relative acc with ITM error', fontsize=titlefont)

ax1.tick_params(axis='both', which='major', labelsize=tickfont)
ax2.tick_params(axis='both', which='major', labelsize=tickfont)
ax3.tick_params(axis='both', which='major', labelsize=tickfont)
ax4.tick_params(axis='both', which='major', labelsize=tickfont)



ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))


# fig.tight_layout()
xaxis = list(range(1, 26))
marker = 'o-'
lw = 2.5
ms = 7
capsize = 5

data = pd.read_excel('multicause_results_plot.xlsx', sheet_name='esd+itm_v3_tV2_plot')

a1 = []
a2 = []
a3 = []
a4 = []
a5 = []

for i in range(25):
      baseline = data['itm0'][i]
      a1.append((data['itm10'][i]-baseline))
      a2.append((data['itm20'][i]-baseline))
      a3.append((data['itm30'][i]-baseline))
      a4.append((data['itm40'][i]-baseline))
      a5.append((data['itm50'][i]-baseline))

print('ave itm acc variation: ', sum(a1)/len(a1), sum(a2)/len(a2), sum(a3)/len(a3), sum(a4)/len(a4),
      sum(a5)/len(a5))


b1 = []
b2 = []
b3 = []
b4 = []

for i in range(25):
      baseline = data['esd0'][i]
      b1.append((data['esd2.5'][i]-baseline))
      b2.append((data['esd5'][i]-baseline))
      b3.append((data['esd7.5'][i]-baseline))
      b4.append((data['esd10'][i]-baseline))

print('ave esd acc variation: ', sum(b1)/len(b1), sum(b2)/len(b2), sum(b3)/len(b3), sum(b4)/len(b4))


ax1.plot(xaxis, data['esd0'], marker, label='noise=0.0%', lw=lw, ms=ms)
ax1.plot(xaxis, data['esd2.5'], marker, label='noise=2.5%', lw=lw, ms=ms)
ax1.plot(xaxis, data['esd5'], marker, label='noise=5.0%', lw=lw, ms=ms)
ax1.plot(xaxis, data['esd7.5'], marker, label='noise=7.5%', lw=lw, ms=ms)
ax1.plot(xaxis, data['esd10'], marker, label='noise=10%', lw=lw, ms=ms)

ax1.set_ylabel("Acc", fontsize=titlefont)
ax1.set_xlabel("No. dataset increment", fontsize=titlefont)

# r'$x_{1}$
ax2.plot(xaxis, data['itm0'], marker, label=r'Added MAE=0', lw=lw, ms=ms)
ax2.plot(xaxis, data['itm10'], marker, label=r'Added MAE=10', lw=lw, ms=ms)
ax2.plot(xaxis, data['itm20'], marker, label=r'Added MAE=20', lw=lw, ms=ms)
ax2.plot(xaxis, data['itm30'], marker, label=r'Added MAE=30', lw=lw, ms=ms)
ax2.plot(xaxis, data['itm40'], marker, label=r'Added MAE=40', lw=lw, ms=ms)
ax2.plot(xaxis, data['itm50'], marker, label=r'Added MAE=50', lw=lw, ms=ms)

ax2.set_ylabel("Acc", fontsize=titlefont)
ax2.set_xlabel("No. dataset increment", fontsize=titlefont)

xaxis_esd = [2.5, 5.0, 7.5, 10.0]
xaxis_itm = [10, 20, 30, 40, 50]

esd_ra = [sum(b1)/len(b1), sum(b2)/len(b2), sum(b3)/len(b3), sum(b4)/len(b4)]
itm_ra = [sum(a1)/len(a1), sum(a2)/len(a2), sum(a3)/len(a3), sum(a4)/len(a4), sum(a5)/len(a5)]

esd_std = [np.std(b1), np.std(b2), np.std(b3), np.std(b4)]
itm_std = [np.std(a1), np.std(a2), np.std(a3), np.std(a4), np.std(a5)]

ax3.xaxis.set_ticks(xaxis_esd)
ax4.xaxis.set_ticks(xaxis_itm)

esd = np.array([b1, b2, b3, b4]).transpose()
itm = np.array([a1, a2, a3, a4, a5]).transpose()


ax3.boxplot(esd, positions=[2.5, 5.0, 7.5, 10.0], widths=0.6)
ax4.boxplot(itm, positions=[10, 20, 30, 40, 50], widths=4)

ax3.set_xlabel("Noise level (%)", fontsize=titlefont)
ax3.set_ylabel("Acc change (%)", fontsize=titlefont)

ax4.set_xlabel("Added MAE", fontsize=titlefont)
ax4.set_ylabel("Acc change (%)", fontsize=titlefont)



leg = ax1.legend()
ax1.legend(fontsize=13, )
leg.get_frame().set_linewidth(0.1)

ax2.legend(fontsize=13,)

plt.savefig('result-noise-tV2.pdf', dpi=600, bbox_inches='tight',pad_inches = 0.05)
plt.show()