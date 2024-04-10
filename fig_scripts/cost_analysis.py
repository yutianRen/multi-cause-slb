import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline


alpha = [i/10 for i in range(1, 11, 2)]
# alpha = [0.5]

p = 0.4
r = 0.09
cm = 0.104

acc = 0.5
beta = [i for i in range(1, 16)]

def calc(acc):
    t_res = {}
    for a in alpha:
        sub_t = []
        for b in beta:
            t = (acc * cm) / (p * r * ((1 + 2* a) * b - acc))
            print(a, b, t,)
            sub_t.append(t)
        t_res[a] = sub_t
    return t_res

font = 16
titlefont = 14
tickfont = 12
legendfont = 13

fig = plt.figure(figsize=(10, 6), facecolor='w', edgecolor='k')
fig.subplots_adjust(left=0.07, bottom=0.096, right=0.98, top=0.94, wspace=0.20, hspace=0.4)

ax1 = plt.subplot(221)
ax2 = plt.subplot(222)
ax3 = plt.subplot(223)
ax4 = plt.subplot(224)

ax1.tick_params(axis='both', which='major', labelsize=tickfont)
ax2.tick_params(axis='both', which='major', labelsize=tickfont)
ax3.tick_params(axis='both', which='major', labelsize=tickfont)
ax4.tick_params(axis='both', which='major', labelsize=tickfont)

ax1.set_title(r'(a) $\Delta acc_{slb} / \Delta acc_{fs}=0.25$', fontsize=titlefont)
ax1.set_ylim(0, 1)
ax1.set_yscale('symlog')
res = calc(0.2)
for key, line in res.items():
    ax1.plot(beta, line, 'o-', label=rf"$\alpha$={key}")
ax1.legend(ncol=2, fontsize=legendfont)
ax1.set_xlabel(r'$\beta$', fontsize=font)
ax1.set_ylabel(r'$t_{compute}$', fontsize=font)


ax2.set_title(r'(b) $\Delta acc_{slb} / \Delta acc_{fs}=0.5$', fontsize=titlefont)
# ax2.set_ylim(0, 10)
ax2.set_yscale('symlog')
res = calc(0.5)
for key, line in res.items():
    ax2.plot(beta, line, 'o-', label=rf"$\alpha$={key}")
ax2.legend(ncol=2, fontsize=legendfont)
ax2.set_xlabel(r'$\beta$', fontsize=font)
ax2.set_ylabel(r'$t_{compute}$', fontsize=font)


ax3.set_title(r'(c) $\Delta acc_{slb} / \Delta acc_{fs}=0.75$', fontsize=titlefont)
# ax3.set_ylim(0, 10)
ax3.set_yscale('symlog')
res = calc(0.75)
for key, line in res.items():
    ax3.plot(beta, line, 'o-', label=rf"$\alpha$={key}")
ax3.legend(ncol=2, fontsize=legendfont)
ax3.set_xlabel(r'$\beta$', fontsize=font)
ax3.set_ylabel(r'$t_{compute}$', fontsize=font)


ax4.set_title(r'(d) $\Delta acc_{slb} / \Delta acc_{fs}=1$', fontsize=titlefont)
ax4.set_yscale('symlog')
res = calc(1)
for key, line in res.items():
    ax4.plot(beta, line, 'o-', label=rf"$\alpha$={key}")
ax4.legend(ncol=2, fontsize=legendfont)
ax4.set_xlabel(r'$\beta$', fontsize=font)
ax4.set_ylabel(r'$t_{compute}$', fontsize=font)

plt.savefig('cost_analysis.pdf', dpi=300, bbox_inches='tight',pad_inches = 0.05)
plt.show()

