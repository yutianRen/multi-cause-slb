import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy

from scipy.optimize import fsolve


plt.rcParams["mathtext.fontset"] = "cm"

def func_solve_tif(tif, y2):
    y1 = 10
    x2 = 100

    ans = y2 - x2 * tif - np.exp(tif) * y1

    return ans


def get_tif_with_error(error):
    max_tif_arr = []
    for x1 in x1_arr:
        y2fs = x2 - np.sqrt(x1 * x2) + y1 * np.sqrt(x2 / x1)
        e_y2fs = error * y2fs

        max_tif = fsolve(func_solve_tif, [0], args=(e_y2fs))
        max_tif_arr.append(max_tif[0])
    return max_tif_arr

x2 = 100
y1 = 10

x1_arr = np.linspace(0.1, x2, 2000)

max_tif_arr = []
true_tif_arr = []

y2fs_error_arr = []
y2slb_arr = []

for x1 in x1_arr:

    true_tif = np.log(np.sqrt(x2 / x1))
    y2slb = x2 * true_tif + np.exp(true_tif) * y1

    # max_tif_arr.append(max_tif)
    true_tif_arr.append(true_tif)
    # y2fs_error_arr.append(e_y2fs)
    y2slb_arr.append(y2slb)
    # print(x1, max_tif, true_tif, e_y2fs, y2slb)


matplotlib.rc('xtick', labelsize=20)
matplotlib.rc('ytick', labelsize=20)

lw = 3

# fig = plt.figure(figsize=(6, 6), facecolor='w', edgecolor='k') # for dx
fig = plt.figure()
fig.subplots_adjust(left=0.12, bottom=0.12, right=0.95, top=0.95, wspace=0.15, hspace=0.15)

ax1 = plt.subplot(111)

ax1.fill_between(x1_arr, get_tif_with_error(1.5), get_tif_with_error(0.5),
                 color='tab:pink', alpha=0.2, label=r'$\epsilon=0.5$')

ax1.fill_between(x1_arr, get_tif_with_error(1.3), get_tif_with_error(0.7),
                 color='tab:green', alpha=0.4, label=r'$\epsilon=0.3$')

ax1.fill_between(x1_arr, get_tif_with_error(1.1), get_tif_with_error(0.9),
                 color='tab:blue', alpha=0.7, label=r'$\epsilon=0.1$')

ax1.plot(x1_arr, true_tif_arr, label=r'$t_{if}$', c='tab:orange')

ax1.set_xlabel(r'$x_{1}$', fontsize=20)
ax1.set_ylabel(r'$t_{if}$', fontsize=20)


ax1.legend()
ax1.legend(fontsize=20)
ax1.grid()

plt.savefig('ds_itm_window-2.pdf', dpi=600, bbox_inches='tight',pad_inches = 0.05)
plt.show()

