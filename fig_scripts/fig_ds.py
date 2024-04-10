import matplotlib
import matplotlib as mpl
import numpy as np
from matplotlib import pyplot as plt

mpl.rcParams['agg.path.chunksize'] = 10000
plt.rcParams["mathtext.fontset"] = "cm"

x2 = 100 #100
y1 = 10

x1 = np.linspace(0.1, x2, 2000)

def y2_slb_error(error_itm, error_y2):
    y2_slb_e = (x2*np.log((x2/x1)**(1/(2 * error_itm))) + y1*(x2/x1)**(1/(2 * error_itm))) * 1 / error_y2
    return y2_slb_e


def y2_slb_error_gen(error_itm, error_y2, a, b):
    if a == 1:
        y2_slb_e = (x2 ** ((1-a)/((a+b)*error_itm) + 1) * x1**((a-1)/((a+b) * error_itm)) *
                    np.log((x2/x1)**(1/((a+b)*error_itm))) + y1 * (x2/x1)**(1/((a+b)*error_itm))) * 1 / error_y2
    else:
        y2_slb_e = (x2 ** ((1-a)/((a+b)*error_itm) + 1) * x1**((a-1)/((a+b) * error_itm)) *
                    (1/(a-1)) * ((x2/x1)**((a-1)/((a+b)*error_itm)) - 1)
                    + y1 * (x2/x1)**(1/((a+b)*error_itm))) * 1 / error_y2

    return y2_slb_e

if __name__ == '__main__':


    # for dx
    matplotlib.rc('xtick', labelsize=20)
    matplotlib.rc('ytick', labelsize=20)
    # fig = plt.figure(figsize=(13, 13), facecolor='w', edgecolor='k') # for dx
    fig = plt.figure(figsize=(6, 10))
    fig.subplots_adjust(left=0.035, bottom=0.05, right=0.98, top=0.96, wspace=0.18, hspace=0.3)

    a = 1
    b = 1

    # DS:
    # no disturbance: x dot = x, y dot = y+x
    # with disturbance: x dot = x+x, y dot = y+x
    if x2 > 0: # for positive systems
        y2_slb = x2*np.log(np.sqrt(x2/x1)) + y1*np.sqrt(x2/x1) # checked

        y2_forw = x2/x1 * (x2-x1+y1) # checked
        y2_trad = x2*np.log(x2/x1) + x2/x1 * y1 # checked
        y2_gt = x2 - np.sqrt(x1*x2) + y1*np.sqrt(x2/x1)
        # y2_gt = x2 + (y1-x1) * np.sqrt(x2/x1) # checked
    else: # for negative systems
        print('negative system')
        y2_slb = np.sqrt(x1 / x2) * (0.5 * np.sqrt(x1 * x2) * (x2 / x1 - 1) + y1)
        y2_forw = x1 / x2 * (1/3 * np.abs(x1) * (np.abs(x1/x2)**(-3) - 1) + y1)
        y2_trad = x1 / x2 * (0.5 * np.abs(x1) * ((x2 / x1)**2 - 1) + y1)
        y2_gt  = np.sqrt(x1 / x2) * (1/3 * np.abs(x1) * (np.abs(x1/x2)**(-3/2) - 1) + y1)


    ax1 = plt.subplot(211)
    lw = 1.6
    ax1.set_yscale('symlog')
    # ax1.set_title(r'$f(x)=x, d(x)=x$', fontsize=24)
    ax1.plot(x1, y2_trad, label='trad', linewidth=lw)
    ax1.plot(x1, y2_gt, label='fs', linewidth=lw)
    ax1.plot(x1, y2_slb, label='slb', linewidth=lw)
    # ax1.plot(x1, y2_forw, label='fwd', linewidth=lw)

    ax1.fill_between(x1, y2_slb_error_gen(1.5, 1, a, b), y2_slb_error_gen(0.5, 1, a, b),
                     color='tab:gray', alpha=0.2, label=r'slb:$|1-\xi_{t}|=0.5$')
    ax1.fill_between(x1, y2_slb_error_gen(1.3, 1, a, b), y2_slb_error_gen(0.7, 1, a, b),
                     color='tab:pink', alpha=0.5, label=r'slb:$|1-\xi_{t}|=0.3$')
    ax1.fill_between(x1, y2_slb_error_gen(1.1, 1, a, b), y2_slb_error_gen(0.9, 1, a, b),
                     color='tab:brown', alpha=0.7, label=r'slb:$|1-\xi_{t}|=0.1$')

    ax1.set_xlabel(r'$x_{1}$', fontsize=24)
    ax1.set_ylabel(r'$y_{2}$', fontsize=24)
    ax1.legend(fontsize=16, ncol=2, columnspacing=0.8)
    ax1.set_title('(a)', fontsize=20)


    ax2 = plt.subplot(212)
    ax2.set_yscale('symlog')
    # ax1.set_title(r'$f(x)=x, d(x)=x$', fontsize=24)
    ax2.plot(x1, y2_trad, label='trad', linewidth=lw)
    ax2.plot(x1, y2_gt, label='fs', linewidth=lw)
    ax2.plot(x1, y2_slb, label='slb', linewidth=lw)
    # ax1.plot(x1, y2_forw, label='fwd', linewidth=lw)


    ax2.fill_between(x1, y2_slb_error_gen(1, 1.5, a, b), y2_slb_error_gen(1, 0.5, a, b),
                     color='tab:gray', alpha=0.2, label=r'slb:$|1-\xi_{e}|=0.5$')
    ax2.fill_between(x1, y2_slb_error_gen(1, 1.3, a, b), y2_slb_error_gen(1, 0.7, a, b),
                     color='tab:pink', alpha=0.5, label=r'slb:$|1-\xi_{e}|=0.3$')
    ax2.fill_between(x1, y2_slb_error_gen(1, 1.1, a, b), y2_slb_error_gen(1, 0.9, a, b),
                     color='tab:brown', alpha=0.7, label=r'slb:$|1-\xi_{e}|=0.1$')


    ax2.set_xlabel(r'$x_{1}$', fontsize=24)
    ax2.set_ylabel(r'$y_{2}$', fontsize=24)
    ax2.legend(fontsize=16, ncol=2, columnspacing=0.8)
    ax2.set_title('(b)', fontsize=20)

    ax1.grid()
    ax2.grid()
    plt.savefig('ds_itm_esd_error-3.pdf', dpi=300, bbox_inches='tight',pad_inches = 0.05)
    plt.show()
