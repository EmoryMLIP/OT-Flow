# plotTraceComparison.py

import matplotlib
try:
    matplotlib.use('TkAgg')                                                                                                                                 
except:
    matplotlib.use('Agg') # for linux server with no tkinter
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import matplotlib.gridspec as gridspec



def plotTraceCompare(domainMiniboone,domainBSDS,domainMNIST,
                     approxTimingMiniboone, approxTimingBDS, approxTimingMNIST,
                     traceErrorMiniboone, traceErrorBDS, traceErrorMNIST,
                     lTimeExact,sPath = "image/traceComparison/"):
    """

    :param domainMiniboone: list, number of hutchinson vectors Miniboone
    :param domainBSDS:                      "                  BSDS300
    :param domainMNIST:                     "                  MNIST
    :param approxTimingMiniboone: list, same length of domain, timings Miniboone
    :param approxTimingBDS:                 "                          BSDS300
    :param approxTimingMNIST:               "                          MNIST
    :param traceErrorMiniboone: list, same length of domain, trace estimation rel. errors Miniboone
    :param traceErrorBDS:                   "                                             BSDS300
    :param traceErrorMNIST:                 "                                             MNIST
    :param lTimeExact: list of 3 timings of exact trace for Miniboone, BSDS, MNIST
    :return: void. plot the figures
    """

    exactTimingMiniboone = lTimeExact[0]  * torch.ones(len(domainMiniboone))
    exactTimingBDS       = lTimeExact[1]  * torch.ones(len(domainBSDS))
    exactTimingMNIST     = lTimeExact[2]  * torch.ones(len(domainMNIST))


    ylim_min = torch.ones(1) * 8e-4
    ylim_max = torch.max(torch.FloatTensor(approxTimingMNIST))
    print("ylim_max = ", ylim_max)

    ylim_min_err = 0.8*torch.min(torch.FloatTensor(traceErrorMNIST))
    ylim_max_err = 1.2*torch.max(torch.FloatTensor(traceErrorMNIST))

    # path to save figure
    if not os.path.exists(os.path.dirname(sPath)):
        os.makedirs(os.path.dirname(sPath))

    # Plots
    fontsize = 20
    title_fontsize = 22

    # we do four plots. the first three share an axis and the last is the relative errors
    fig = plt.figure()
    plt.clf()
    fig.set_size_inches(20, 4.1)
    outer = gridspec.GridSpec(1, 2, wspace=0.20, width_ratios= [2.7,1.0])
    # the first three that I want to share an axis
    inner = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=outer[0], wspace=0.08)

    # yticks values
    rangeAll = [10**(-3),10**(-2),10**(-1),1]


    # fig1 Miniboone timings
    ax = plt.Subplot(fig, inner[0])
    ax.semilogy(domainMiniboone, exactTimingMiniboone, linewidth=3, linestyle='dashed', color='black')
    ax.semilogy(domainMiniboone, approxTimingMiniboone, marker="o", markersize=12, color='tab:blue')
    ax.set_xticks(domainMiniboone)
    ax.set_yticks(rangeAll)
    ax.set_ylabel("Runtime (s)", fontsize=title_fontsize)
    ax.set_ylim((ylim_min, ylim_max))
    # ax.set_xlabel("# of Hutchinson vectors", fontsize=title_fontsize)
    ax.set_title("(a) MINIBOONE, d=43", fontsize=title_fontsize)
    # try to force tick font o be large
    ax.tick_params(labelsize=fontsize, which='both', direction='in')
    fig.add_subplot(ax)


    # fig2 BSDS300 timings
    ax = plt.Subplot(fig, inner[1])
    ax.semilogy(domainBSDS, exactTimingBDS, linewidth=3, linestyle='dashed', color='black')
    ax.semilogy(domainBSDS, approxTimingBDS, marker=">", markersize=12, color='tab:green')
    ax.set_xticks(domainBSDS)
    ax.set_yticks(rangeAll)
    ax.set_ylim((ylim_min, ylim_max))
    ax.set_xlabel("Number of Hutchinson Vectors", fontsize=title_fontsize)
    ax.set_title("(b) BSDS300, d=63", fontsize=title_fontsize)
    # try to force tick's font size to be large
    ax.tick_params(labelsize=fontsize, which='both', direction='in')
    ax.tick_params(left=True, labelleft=False)
    fig.add_subplot(ax)



    # fig3 MNIST timings
    ax = plt.Subplot(fig, inner[2])
    ax.semilogy(domainMNIST, exactTimingMNIST, linewidth=3, linestyle='dashed', color='black')
    ax.semilogy(domainMNIST, approxTimingMNIST, marker="x", markersize=12, color='tab:red')
    ax.set_xticks([1, 200, 400, 600, 784])
    ax.set_yticks(rangeAll)
    ax.set_ylim((ylim_min, ylim_max))
    # ax.set_xlabel("# of Hutchinson vectors", fontsize=title_fontsize)
    ax.set_title("(c) MNIST, d=784", fontsize=title_fontsize)
    # try to force tick's font size to be large
    ax.tick_params(labelsize=fontsize, which='both', direction='in')
    ax.tick_params(left=True, labelleft=False)
    fig.add_subplot(ax)

    # fig 4 relative errors
    ax = plt.Subplot(fig, outer[1])
    ax.semilogy(domainMiniboone, traceErrorMiniboone, marker="o", markersize=12, color='tab:blue')
    ax.semilogy(domainBSDS, traceErrorBDS, marker=">", markersize=12, color='tab:green')
    ax.semilogy(domainMNIST, traceErrorMNIST, marker="x", markersize=12, color='tab:red')
    # fake line to add to the legend
    ax.plot(domainMNIST, 1e-16*torch.ones(len(domainMNIST)), linewidth=3, linestyle='dashed', color='black')
    ax.set_ylim((ylim_min_err, ylim_max_err))
    ax.set_ylabel("Relative Error", fontsize=title_fontsize)
    ax.set_xlabel("Number of Hutchinson Vectors", fontsize=title_fontsize)
    ax.legend(['Hutchinson d=43', 'Hutchinson d=63', 'Hutchinson d=784', 'Exact'],fontsize=fontsize,bbox_to_anchor=(0.45,0.3))
    ax.set_title("(d) Accuracy of Estimators", fontsize=title_fontsize)
    ax.tick_params(labelsize=fontsize, which='both', direction='in')
    fig.add_subplot(ax)

    plt.subplots_adjust(top=0.9,bottom=0.2)


    # plt.show()
    fig.savefig(sPath+'all4Trace.pdf', dpi=600)
    print("figure saved to ", sPath+'all4Trace.pdf')


if __name__ == '__main__':

    # results from our runs. Values are an average from 20 runs.

    # Miniboone:
    domainMiniboone = [1, 10, 20, 30, 43]

    approxTimingMiniboone =  [0.0020979393739253283, 0.01768721267580986, 0.03584431856870651, 0.05631520226597786, 0.07971490919589996]

    traceErrorMiniboone   =  [0.21124808490276337, 0.06990428268909454, 0.04749888554215431, 0.04056185111403465, 0.030909394845366478]

    exactTimingMiniboone  =  0.0020757552206516266

    # BSDS
    domainBSDS = [1, 10, 20, 30, 40, 50, 63]

    approxTimingBDS = [0.0021919216960668564, 0.018128778785467148, 0.03683362901210785, 0.05762229114770889,
              0.07594551891088486, 0.09460346400737762, 0.12033005058765411]

    traceErrorBDS = [0.22224731743335724, 0.0712037906050682, 0.05391544848680496, 0.04101308062672615, 0.03516305238008499,
            0.03153880313038826, 0.028710657730698586]

    exactTimingBDS = 0.00223581600189209

    # MNIST
    domainMNIST = [1, 100, 200, 300, 400, 500, 600, 700, 784]

    approxTimingMNIST = [0.002257217885926366, 0.19411349296569824, 0.39349645376205444, 0.6066120266914368, 0.8377354741096497,
              1.1355806589126587, 1.366000771522522, 1.6045491695404053, 1.7919518947601318]

    traceErrorMNIST = [0.2409706562757492, 0.02531665936112404, 0.01699836179614067, 0.012928245589137077, 0.012266159988939762,
            0.010199744254350662, 0.00982920452952385, 0.00876554287970066, 0.008366582915186882]

    exactTimingMNIST =  0.0056630992174148565


    lTimeExact = [exactTimingMiniboone, exactTimingBDS, exactTimingMNIST]

    plotTraceCompare(domainMiniboone,domainBSDS,domainMNIST,
                     approxTimingMiniboone, approxTimingBDS, approxTimingMNIST,
                     traceErrorMiniboone, traceErrorBDS, traceErrorMNIST,
                     lTimeExact)




