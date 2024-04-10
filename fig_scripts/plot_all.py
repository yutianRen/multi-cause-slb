import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import make_interp_spline
from matplotlib.ticker import FormatStrFormatter
import pandas as pd
import seaborn as sns

fig = plt.figure(figsize=(25, 16), facecolor='w', edgecolor='k')
fig.subplots_adjust(left=0.05, bottom=0.05, right=0.98, top=0.98, wspace=0.1, hspace=0.4)

ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)


titlefont = 28
tickfont = 26

ax1.tick_params(axis='both', which='major', labelsize=tickfont)
ax2.tick_params(axis='both', which='major', labelsize=tickfont)
ax3.tick_params(axis='both', which='major', labelsize=tickfont)
ax4.tick_params(axis='both', which='major', labelsize=tickfont)


ax1.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax2.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax3.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
ax4.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))


# fig.tight_layout()
xaxis = list(range(0, 26))
marker = 'o-'
lw = 2
ms = 10
capsize = 5
errorevery=2

data = pd.read_excel('multicause_results_plot.xlsx', sheet_name='mean_v3_tV2_plot')
data_std = pd.read_excel('multicause_results_plot.xlsx', sheet_name='std_v3_tV2_plot')
print(data['nwslb'].to_numpy().shape, data_std['nwslb'].shape)


# results_x0.05z0.05t1_b75f25h1020
# h20
ax1.set_title('(a) no wind', fontsize=titlefont) #'$x_{1}$'
ax1.errorbar(xaxis, data['nwslb'], data_std['nwslb'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='SLB-25', lw=lw, ms=ms)
ax1.errorbar(xaxis, data['nw50'], data_std['nw50'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='SLB-5', lw=lw, ms=ms)
ax1.errorbar(xaxis, data['nw75'], data_std['nw75'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='SLB-75', lw=lw, ms=ms)

ax1.errorbar(xaxis, data['nwfs'], data_std['nwfs'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='FS', lw=lw, ms=ms)
ax1.errorbar(xaxis, data['nwpl'], data_std['nwpl'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='Pseudo', lw=lw, ms=ms)
ax1.errorbar(xaxis, data['nwmix'], data_std['nwmix'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='Mixmatch', lw=lw, ms=ms)
ax1.errorbar(xaxis, data['nwpf'], data_std['nwpf'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='PseudoFlex', lw=lw, ms=ms)
ax1.errorbar(xaxis, data['nwsf'], data_std['nwsf'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='SoftMatch', lw=lw, ms=ms)
ax1.errorbar(xaxis, data['nwfr'], data_std['nwfr'], c='cadetblue', fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='FreeMatch', lw=lw, ms=ms)

ax1.set_xlabel('Data', fontsize=tickfont)
ax1.set_ylabel('Acc (%)', fontsize=tickfont)

# results_x0.05z0.05t1_b75f25h1025
# h25
ax2.set_title('(b) $wind=(-0.5,-0.5,-0.5))$', fontsize=titlefont)
ax2.errorbar(xaxis, data['w0.5slb'], data_std['w0.5slb'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='SLB-25', lw=lw, ms=ms) # 250, 0.1
ax2.errorbar(xaxis, data['w0.550'], data_std['w0.550'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='SLB-50', lw=lw, ms=ms)
ax2.errorbar(xaxis, data['w0.575'], data_std['w0.575'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='SLB-75', lw=lw, ms=ms)

ax2.errorbar(xaxis, data['w0.5fs'], data_std['w0.5fs'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='FS', lw=lw, ms=ms)
ax2.errorbar(xaxis, data['w0.5pl'], data_std['w0.5pl'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='PseudoLabel', lw=lw, ms=ms)
ax2.errorbar(xaxis, data['w0.5mix'], data_std['w0.5mix'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='MixMatch', lw=lw, ms=ms)
# ax2.errorbar(xaxis, data['w0.5fix'], marker, label='FixMatch', lw=lw, ms=ms)
# ax2.errorbar(xaxis, data['w0.5flex'], marker, label='FlexMatch', lw=lw, ms=ms)
ax2.errorbar(xaxis, data['w0.5pf'], data_std['w0.5pf'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='PseudoFlexLabel', lw=lw, ms=ms)
# ax2.errorbar(xaxis, data['w0.5sm'], marker, label='SimMatch', lw=lw, ms=ms)
ax2.errorbar(xaxis, data['w0.5sf'], data_std['w0.5sf'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='SoftMatch', lw=lw, ms=ms)
ax2.errorbar(xaxis, data['w0.5fr'], data_std['w0.5fr'], c='cadetblue', fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='FreeMatch', lw=lw, ms=ms)

ax2.set_xlabel('Data', fontsize=tickfont)
ax2.set_ylabel('Acc (%)', fontsize=tickfont)

# results_x0.05z0.05t1_b25f25h1015
# b25
ax3.set_title('(c) $wind=(-1,-1,-1)$', fontsize=titlefont)
ax3.errorbar(xaxis, data['w1.0slb'], data_std['w1.0slb'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='SLB-25', solid_joinstyle='bevel', lw=lw, ms=ms)
ax3.errorbar(xaxis, data['w1.050'], data_std['w1.050'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='SLB-50', lw=lw, ms=ms)
ax3.errorbar(xaxis, data['w1.075'], data_std['w1.075'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='SLB-75', lw=lw, ms=ms)

ax3.errorbar(xaxis, data['w1.0fs'], data_std['w1.0fs'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='FS', solid_joinstyle='bevel', lw=lw, ms=ms)
ax3.errorbar(xaxis, data['w1.0pl'], data_std['w1.0pl'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='Pseudo', solid_joinstyle='bevel', lw=lw, ms=ms)
ax3.errorbar(xaxis, data['w1.0mix'], data_std['w1.0mix'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='Mixmatch', lw=lw, ms=ms)
# ax3.errorbar(xaxis, data['w1.0fix'], marker, label='Fixmatch', solid_joinstyle='bevel', lw=lw, ms=ms)
# ax3.errorbar(xaxis, data['w1.0flex'], marker, label='Flexmatch', solid_joinstyle='bevel', lw=lw, ms=ms)
ax3.errorbar(xaxis, data['w1.0pf'], data_std['w1.0pf'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='PseudoFlex', solid_joinstyle='bevel', lw=lw, ms=ms)
# ax3.errorbar(xaxis, data['w1.0sm'], marker, label='SimMatch', solid_joinstyle='bevel', lw=lw, ms=ms)
ax3.errorbar(xaxis, data['w1.0sf'], data_std['w1.0sf'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='SoftMatch', solid_joinstyle='bevel', lw=lw, ms=ms)
ax3.errorbar(xaxis, data['w1.0fr'], data_std['w1.0fr'], c='cadetblue', fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='FreeMatch', solid_joinstyle='bevel', lw=lw, ms=ms)

ax3.set_xlabel('Data', fontsize=tickfont)
ax3.set_ylabel('Acc (%)', fontsize=tickfont)

# results_x0.05z0.05t1_b50f25h1015
# b50
ax4.set_title('(d) $wind=(-1.5,-1.5,-1.5)$', fontsize=titlefont)
ax4.errorbar(xaxis, data['w1.5slb'], data_std['w1.5slb'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='SLB-25', lw=lw, ms=ms)
ax4.errorbar(xaxis, data['w1.550'], data_std['w1.550'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='SLB-50', lw=lw, ms=ms)
ax4.errorbar(xaxis, data['w1.575'], data_std['w1.575'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='SLB-75', lw=lw, ms=ms)

ax4.errorbar(xaxis, data['w1.5fs'], data_std['w1.5fs'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='FS', lw=lw, ms=ms)
ax4.errorbar(xaxis, data['w1.5pl'], data_std['w1.5pl'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='PseudoLabel', lw=lw, ms=ms)
ax4.errorbar(xaxis, data['w1.5mix'], data_std['w1.5mix'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='Mixmatch', lw=lw, ms=ms)
# ax4.errorbar(xaxis, data['w1.5fix'], marker, label='Fixmatch', lw=lw, ms=ms)
# ax4.errorbar(xaxis, data['w1.5flex'], marker, label='Flexmatch', lw=lw, ms=ms)
ax4.errorbar(xaxis, data['w1.5pf'], data_std['w1.5pf'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='PseudoLabelFlex', lw=lw, ms=ms)
# ax4.errorbar(xaxis, data['w1.5sm'], marker, label='SimMatch', lw=lw, ms=ms)
ax4.errorbar(xaxis, data['w1.5sf'], data_std['w1.5sf'], fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='SoftMatch', lw=lw, ms=ms)
ax4.errorbar(xaxis, data['w1.5fr'], data_std['w1.5fr'], c='cadetblue', fmt=marker, markevery=errorevery, errorevery=errorevery, capsize=capsize, label='FreeMatch', lw=lw, ms=ms)

ax4.set_xlabel('Data', fontsize=tickfont)
ax4.set_ylabel('Acc (%)', fontsize=tickfont)


leg = ax2.legend()
ax2.legend(bbox_to_anchor=(0.65, -0.1), fontsize=24, ncol=5, frameon=False)
leg.get_frame().set_linewidth(0.1)
plt.savefig('result-all-2.pdf', dpi=600, bbox_inches='tight',pad_inches = 0.05)
plt.show()