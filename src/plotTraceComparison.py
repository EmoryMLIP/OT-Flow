# plotTraceComparison.py

try:
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('agg') # for linux server with no tkinter
    import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import matplotlib.gridspec as gridspec
import sklearn



def bootstrap(values, nIter, alpha):
    """
    bootstrapping to create error bounds, uses resmapling with replacement of sample size n-4
    :param values: n-by-m matrix, n = number of runs to resmaple from, m = observations per run
    :param nIter: int, number of resamples
    :param alpha: float, percentile bounds
    :return: lower bounds, mean, upper bounds
    """
    p1 = ((1.0 - alpha) / 2.0) * 100
    p2 = (alpha + ((1.0 - alpha) / 2.0)) * 100
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha * 100, p1, p2))

    stats = list()
    nSize = values.shape[0] - 4
    for i in range(nIter):
        sample = sklearn.utils.resample(values, n_samples=nSize)
        stats.append(np.mean(sample, 0))
    stats = np.array(stats)
    lower = np.percentile(stats, p1, axis=0)
    upper = np.percentile(stats, p2, axis=0)
    avg  = np.mean(values, axis=0)
    print('lower :', lower)
    print('mean  :', avg)
    print('upper :', upper)
    return lower, avg, upper
    return lower, avg, upper




def plotTraceCompare(domainMiniboone,domainBSDS,domainMNIST,
                     approxTimingMiniboone, approxTimingBDS, approxTimingMNIST,
                     traceErrorMiniboone, traceErrorBDS, traceErrorMNIST,
                     lTimeExact,sPath = "../image/traceComparison/", bErrBar=False):
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
    :param bErrBar:    boolean, True means to plot the error bounds
    :return: void. plot the figures
    """

    shade=0.3

    exactTimingMiniboone = lTimeExact[0]  * torch.ones(len(domainMiniboone))
    exactTimingBDS       = lTimeExact[1]  * torch.ones(len(domainBSDS))
    exactTimingMNIST     = lTimeExact[2]  * torch.ones(len(domainMNIST))

    exactTimingMiniboone = exactTimingMiniboone.cpu().detach().numpy()
    approxTimingMiniboone= approxTimingMiniboone.cpu().detach().numpy()
    traceErrorMiniboone  = traceErrorMiniboone.cpu().detach().numpy()
    exactTimingBDS       = exactTimingBDS.cpu().detach().numpy()
    approxTimingBDS      = approxTimingBDS.cpu().detach().numpy()
    traceErrorBDS        = traceErrorBDS.cpu().detach().numpy()
    exactTimingMNIST     = exactTimingMNIST.cpu().detach().numpy()
    approxTimingMNIST    = approxTimingMNIST.cpu().detach().numpy()
    traceErrorMNIST      = traceErrorMNIST.cpu().detach().numpy()


    if bErrBar:
        # calculate error bars by bootstrapping
        # compute mean of 'nIter' samples with replacement of size n-4
        nIter = 4000
        alpha = 0.99
        # miniboone
        exactMiniLower , exactTimingMiniboone , exactMiniUpper  = bootstrap(exactTimingMiniboone, nIter, alpha)
        approxMiniLower, approxTimingMiniboone, approxMiniUpper = bootstrap(approxTimingMiniboone, nIter, alpha)
        errMiniLower   , traceErrorMiniboone  , errMiniUpper    = bootstrap(traceErrorMiniboone, nIter, alpha)
        # BSDS
        exactBSDSLower , exactTimingBDS , exactBSDSUpper  = bootstrap(exactTimingBDS, nIter, alpha)
        approxBSDSLower, approxTimingBDS, approxBSDSUpper = bootstrap(approxTimingBDS, nIter, alpha)
        errBSDSLower   , traceErrorBDS  , errBSDSUpper    = bootstrap(traceErrorBDS, nIter, alpha)
        # MNIST
        exactMNISTLower , exactTimingMNIST , exactMNISTUpper  = bootstrap(exactTimingMNIST, nIter, alpha)
        approxMNISTLower, approxTimingMNIST, approxMNISTUpper = bootstrap(approxTimingMNIST, nIter, alpha)
        errMNISTLower   , traceErrorMNIST  , errMNISTUpper    = bootstrap(traceErrorMNIST, nIter, alpha)

    else:
        # just calculate the mean
        exactTimingMiniboone  = np.mean( exactTimingMiniboone , axis=0)
        approxTimingMiniboone = np.mean(approxTimingMiniboone, axis=0)
        traceErrorMiniboone   = np.mean(traceErrorMiniboone, axis=0)
        exactTimingBDS        = np.mean(exactTimingBDS, axis=0)
        approxTimingBDS       = np.mean(approxTimingBDS, axis=0)
        traceErrorBDS         = np.mean(traceErrorBDS, axis=0)
        exactTimingMNIST      = np.mean(exactTimingMNIST, axis=0)
        approxTimingMNIST     = np.mean(approxTimingMNIST, axis=0)
        traceErrorMNIST       = np.mean(traceErrorMNIST, axis=0)


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
    if bErrBar:
        ax.fill_between(domainMiniboone, exactMiniLower, exactMiniUpper, alpha=shade, color='black') # for exact
        ax.fill_between(domainMiniboone, approxMiniLower, approxMiniUpper, alpha=shade, color='tab:blue')  # for hutch
    ax.semilogy(domainMiniboone, exactTimingMiniboone, linewidth=3, linestyle='dashed', color='black')
    ax.semilogy(domainMiniboone, approxTimingMiniboone, marker="o", markersize=12,  linestyle=':', color='tab:blue')
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
    if bErrBar:
        ax.fill_between(domainBSDS, exactBSDSLower, exactBSDSUpper, alpha=shade, color='black') # for exact
        ax.fill_between(domainBSDS, approxBSDSLower, approxBSDSUpper, alpha=shade, color='tab:green')  # for hutch
    ax.semilogy(domainBSDS, exactTimingBDS, linewidth=3, linestyle='dashed', color='black')
    ax.semilogy(domainBSDS, approxTimingBDS, marker=">", markersize=12,  linestyle=':', color='tab:green')
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
    if bErrBar:
        ax.fill_between(domainMNIST, exactMNISTLower, exactMNISTUpper, alpha=shade, color='black') # for exact
        ax.fill_between(domainMNIST, approxMNISTLower, approxMNISTUpper, alpha=shade, color='tab:red')  # for hutch
    ax.semilogy(domainMNIST, exactTimingMNIST, linewidth=3, linestyle='dashed', color='black')
    ax.semilogy(domainMNIST, approxTimingMNIST, marker="x", markersize=12, linestyle=':', color='tab:red')
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
    if bErrBar:
        ax.fill_between(domainMiniboone, errMiniLower , errMiniUpper,  color='tab:blue', alpha=shade)
        ax.fill_between(domainBSDS     , errBSDSLower , errBSDSUpper,  color='tab:green', alpha=shade)
        ax.fill_between(domainMNIST    , errMNISTLower, errMNISTUpper, color='tab:red', alpha=shade)
    ax.semilogy(domainMiniboone, traceErrorMiniboone, marker="o", markersize=12, linestyle=':',color='tab:blue')
    ax.semilogy(domainBSDS, traceErrorBDS, marker=">", markersize=12, linestyle=':', color='tab:green')
    ax.semilogy(domainMNIST, traceErrorMNIST, marker="x", markersize=12, linestyle=':', color='tab:red')
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

    approxTimingMiniboone = torch.tensor([[0.00204698, 0.02248192, 0.05925171, 0.09269555, 0.13181952],
        [0.00197325, 0.02188800, 0.05658624, 0.08444519, 0.13243289],
        [0.00198451, 0.02560000, 0.05685863, 0.09898496, 0.13945447],
        [0.00199782, 0.02196685, 0.05347942, 0.09409127, 0.13435186],
        [0.00207565, 0.02258125, 0.05532877, 0.09219276, 0.13214721],
        [0.00200294, 0.02304615, 0.06069043, 0.09288090, 0.13366579],
        [0.00198656, 0.02010317, 0.05643674, 0.09239347, 0.13222298],
        [0.00201728, 0.02313114, 0.05497242, 0.09327411, 0.13496320],
        [0.00195686, 0.02510848, 0.05944423, 0.09257779, 0.12938856],
        [0.00200090, 0.01983590, 0.05545882, 0.09132032, 0.13388595],
        [0.00230605, 0.02441216, 0.06083686, 0.09413837, 0.13533390],
        [0.00202854, 0.02355405, 0.05499494, 0.09336935, 0.13517620],
        [0.00253235, 0.02976358, 0.05309030, 0.09212723, 0.13202228],
        [0.00200090, 0.02030694, 0.05293158, 0.09495040, 0.13553973],
        [0.00194970, 0.02337178, 0.05714330, 0.09082573, 0.13207449],
        [0.00200499, 0.01968742, 0.05493453, 0.09481011, 0.14179942],
        [0.00207258, 0.02444391, 0.05693235, 0.09386291, 0.13639885],
        [0.00216883, 0.02145178, 0.05614080, 0.09291776, 0.13407233],
        [0.00296346, 0.02311270, 0.05964186, 0.09435239, 0.13308007],
        [0.00210944, 0.02242970, 0.06016410, 0.09175961, 0.13343847]])

    traceErrorMiniboone   =  torch.tensor([[0.22399592, 0.07008030, 0.05251038, 0.04112926, 0.03358838],
        [0.21175259, 0.06972103, 0.04838064, 0.04045253, 0.03306658],
        [0.21496193, 0.06821524, 0.04756567, 0.03840676, 0.03136125],
        [0.19982451, 0.06403704, 0.04720526, 0.03827408, 0.03041435],
        [0.20888096, 0.06603654, 0.04901867, 0.03706970, 0.03287060],
        [0.21129309, 0.06779046, 0.04861399, 0.03830324, 0.03135169],
        [0.20985791, 0.06680115, 0.04792466, 0.03696685, 0.03245528],
        [0.21677414, 0.06837571, 0.04581433, 0.04125229, 0.03183835],
        [0.21320291, 0.07022005, 0.04878217, 0.04066725, 0.03123112],
        [0.21459921, 0.07032831, 0.04619167, 0.04150819, 0.03356071],
        [0.22348940, 0.06709059, 0.04981212, 0.04132729, 0.03208493],
        [0.21790127, 0.06918178, 0.04933133, 0.03818259, 0.03425195],
        [0.22167198, 0.06969723, 0.04703050, 0.03707674, 0.03329653],
        [0.24082674, 0.07491565, 0.04908932, 0.04273553, 0.03606861],
        [0.19499324, 0.06544592, 0.04834570, 0.04035131, 0.03164196],
        [0.23307209, 0.07598016, 0.05314977, 0.04511942, 0.03460030],
        [0.19148827, 0.06681481, 0.04595243, 0.03640851, 0.03197488],
        [0.19533448, 0.06416275, 0.04717834, 0.03703568, 0.03247492],
        [0.22210748, 0.06772004, 0.04818948, 0.04124676, 0.03337538],
        [0.20792861, 0.06343383, 0.04265359, 0.03540466, 0.03049848]])

    exactTimingMiniboone  =  torch.tensor([[0.00189805], [0.00188099], [0.00189171], [0.00211219], [0.00200957],
        [0.00199389], [0.00214022],  [0.00192339], [0.00192688], [0.00191530], [0.00189725], [0.00189731],
        [0.00195821], [0.00210774],  [0.00200566], [0.00228646], [0.00196464], [0.00193050], [0.00244557],
        [0.00233907]])


    # BSDS
    domainBSDS = [1, 10, 20, 30, 40, 50, 63]

    approxTimingBDS = torch.tensor([[0.00196096, 0.01992499, 0.05914829, 0.09389056, 0.12557927, 0.15675801, 0.19541503],
        [0.00237568, 0.02242867, 0.05595239, 0.08900916, 0.12594381, 0.15561421, 0.19873485],
        [0.00209510, 0.02482176, 0.05906535, 0.09127732, 0.12411802, 0.15319961, 0.19669504],
        [0.00195174, 0.02213274, 0.05481574, 0.09024819, 0.12026573, 0.15681741, 0.19851059],
        [0.00200602, 0.02337587, 0.05639168, 0.09198797, 0.12295783, 0.15485133, 0.19691212],
        [0.00244122, 0.02244096, 0.06035251, 0.09647001, 0.12460339, 0.15235789, 0.19470440],
        [0.00245248, 0.02241946, 0.05773619, 0.09608295, 0.12370022, 0.15242343, 0.19164263],
        [0.00202035, 0.02782413, 0.06032794, 0.09088410, 0.12083405, 0.15557325, 0.19412991],
        [0.00268902, 0.02401485, 0.06104678, 0.09072845, 0.11465625, 0.15696485, 0.19905843],
        [0.00197837, 0.02306765, 0.05657805, 0.09488384, 0.11945062, 0.15574835, 0.19231643],
        [0.00201626, 0.02254848, 0.05652992, 0.08919347, 0.12453479, 0.15620403, 0.19640626],
        [0.00205107, 0.02238976, 0.06089932, 0.09200948, 0.12544614, 0.15656449, 0.19646567],
        [0.00204902, 0.02313216, 0.05572813, 0.09355161, 0.12560281, 0.15204352, 0.19627623],
        [0.00199168, 0.02330726, 0.05735526, 0.09469235, 0.12275814, 0.15653887, 0.19439821],
        [0.00218010, 0.02524467, 0.05821849, 0.09218150, 0.12721151, 0.15646823, 0.20083918],
        [0.00196403, 0.02051174, 0.05868442, 0.09549414, 0.13213082, 0.15206195, 0.20078899],
        [0.00204288, 0.02245222, 0.05068083, 0.09364992, 0.12617932, 0.15401575, 0.19684762],
        [0.00255693, 0.02551603, 0.05791744, 0.09331609, 0.12521063, 0.15728435, 0.19802830],
        [0.00206643, 0.02171597, 0.05827277, 0.08563917, 0.12383948, 0.15793253, 0.19470029],
        [0.00250368, 0.02256077, 0.05704192, 0.09220915, 0.13220045, 0.15089153, 0.19225907]])

    traceErrorBDS = torch.tensor([[0.22224732, 0.07398839, 0.04687803, 0.04231799, 0.03719367, 0.03312020, 0.02866898],
        [0.22899885, 0.06637849, 0.05120244, 0.03810278, 0.03313917, 0.02972283, 0.02768398],
        [0.22437161, 0.06693737, 0.04891766, 0.04001211, 0.03618139, 0.03179351, 0.02886512],
        [0.23279119, 0.06872935, 0.04922526, 0.03858438, 0.03428061, 0.03056928, 0.02815522],
        [0.22483462, 0.06861812, 0.04778925, 0.04283748, 0.03553044, 0.03134139, 0.02784217],
        [0.24206249, 0.07523844, 0.05557442, 0.04315356, 0.03542253, 0.03241287, 0.02874050],
        [0.21107937, 0.06687794, 0.04701358, 0.04142660, 0.03364031, 0.03075318, 0.02824883],
        [0.22436637, 0.06825909, 0.04680301, 0.03896724, 0.03722268, 0.03046616, 0.02864970],
        [0.22186267, 0.06819139, 0.04920222, 0.03770087, 0.03415415, 0.03086751, 0.02698998],
        [0.22473182, 0.06998120, 0.04632138, 0.04015036, 0.03232786, 0.03110034, 0.02825256],
        [0.20607713, 0.06801096, 0.04883099, 0.03788529, 0.03505293, 0.03012344, 0.02555213],
        [0.21236718, 0.07406570, 0.04891364, 0.03851217, 0.03553124, 0.03287421, 0.02734543],
        [0.21273059, 0.06926261, 0.04765013, 0.04096797, 0.03521183, 0.03151200, 0.02731446],
        [0.20699681, 0.06645167, 0.05195054, 0.03938620, 0.03577903, 0.03134636, 0.02641797],
        [0.21683121, 0.06875283, 0.04659092, 0.03796173, 0.03430184, 0.03016153, 0.02521137],
        [0.23120171, 0.07039053, 0.05147018, 0.03970526, 0.03499187, 0.03093808, 0.02673643],
        [0.22178663, 0.07050845, 0.04779807, 0.03904643, 0.03483230, 0.03055800, 0.02810326],
        [0.23234105, 0.07264428, 0.04754635, 0.04069406, 0.03513086, 0.03031974, 0.02696989],
        [0.23352592, 0.07050001, 0.05279808, 0.04384740, 0.03565584, 0.03113800, 0.02937748],
        [0.22431205, 0.06739795, 0.04708927, 0.04052063, 0.03416919, 0.03165910, 0.02772815]])

    exactTimingBDS = torch.tensor([[0.00243334], [0.00240138], [0.00223862], [0.00191088], [0.00186822], [0.00238966],
                             [0.00229366], [0.00193875], [0.00300925], [0.00189766], [0.00211056], [0.00193008],
                             [0.00190336], [0.00192000], [0.00198621], [0.00190147], [0.00194582], [0.00254944],
                             [0.00196250], [0.00239798]])

    # MNIST
    domainMNIST = [1, 100, 200, 300, 400, 500, 600, 700, 784]

    approxTimingMNIST = torch.tensor([[2.08383985e-03, 3.03277105e-01, 6.34796023e-01, 9.73454297e-01,
         1.26875138e+00, 1.58261967e+00, 1.90680981e+00, 2.20069671e+00, 2.48040366e+00],
        [2.18419195e-03, 3.02417904e-01, 6.34218514e-01, 9.57537293e-01, 1.27308381e+00,
         1.60399556e+00, 1.89956093e+00, 2.21912169e+00, 2.48672962e+00],
        [2.16268795e-03, 3.08395028e-01, 6.40161812e-01, 9.49754894e-01, 1.25998187e+00,
         1.59414876e+00, 1.91681635e+00, 2.22077632e+00, 2.48915648e+00],
        [2.31219199e-03, 2.99216896e-01, 6.27590120e-01, 9.52667117e-01, 1.26950192e+00,
         1.59411609e+00, 1.88158262e+00, 2.22522688e+00, 2.48260713e+00],
        [2.14118417e-03, 3.02231610e-01, 6.31080925e-01, 9.50872004e-01, 1.27247977e+00,
         1.58668697e+00, 1.88744187e+00, 2.23091602e+00, 2.47214985e+00],
        [2.04185606e-03, 3.02155763e-01, 6.25741839e-01, 9.47705865e-01, 1.27614772e+00,
         1.58635414e+00, 1.90940773e+00, 2.22117901e+00, 2.51428246e+00],
        [2.44326401e-03, 3.10180873e-01, 6.38756812e-01, 9.42061603e-01, 1.27602589e+00,
         1.57963061e+00, 1.91455340e+00, 2.22590876e+00, 2.46318984e+00],
        [2.09100801e-03, 3.21886212e-01, 6.28022254e-01, 9.50401008e-01, 1.27358270e+00,
         1.59494138e+00, 1.91619575e+00, 2.22641444e+00, 2.50357032e+00],
        [2.26099207e-03, 3.01608980e-01, 6.25809371e-01, 9.56489742e-01, 1.26890695e+00,
         1.56973970e+00, 1.90321040e+00, 2.20976448e+00, 2.48652816e+00],
        [2.08998402e-03, 3.05045515e-01, 6.38055444e-01, 9.51280653e-01, 1.27333987e+00,
         1.57056820e+00, 1.92144990e+00, 2.23278165e+00, 2.49454784e+00],
        [2.05004821e-03, 3.10762554e-01, 6.33757710e-01, 9.59020019e-01, 1.25874281e+00,
         1.59699655e+00, 1.90024710e+00, 2.21750259e+00, 2.47988105e+00],
        [2.14630389e-03, 3.16698641e-01, 6.24750614e-01, 9.55690980e-01, 1.27454925e+00,
         1.59527731e+00, 1.91894329e+00, 2.22634172e+00, 2.50078821e+00],
        [2.00806395e-03, 3.14042360e-01, 6.37971461e-01, 9.40902412e-01, 1.28086519e+00,
         1.59389281e+00, 1.91719842e+00, 2.23712349e+00, 2.49246931e+00],
        [1.99577608e-03, 3.02376956e-01, 6.46013975e-01, 9.51083004e-01, 1.28144896e+00,
         1.60936344e+00, 1.91367579e+00, 2.23405361e+00, 2.50937223e+00],
        [3.06073599e-03, 3.16940308e-01, 6.41007602e-01, 9.51192558e-01, 1.27902007e+00,
         1.58712018e+00, 1.92633653e+00, 2.21752119e+00, 2.50338197e+00],
        [2.09817593e-03, 3.12108040e-01, 6.56091154e-01, 9.85183239e-01, 1.28402126e+00,
         1.61969662e+00, 1.90231442e+00, 2.23895645e+00, 2.51221514e+00],
        [2.06336007e-03, 3.09849083e-01, 6.33301973e-01, 9.46757615e-01, 1.27894533e+00,
         1.57923329e+00, 1.91604125e+00, 2.21328068e+00, 2.50347829e+00],
        [2.06336007e-03, 3.05488884e-01, 6.38731241e-01, 9.60516095e-01, 1.25757539e+00,
         1.60113358e+00, 1.90383101e+00, 2.24165583e+00, 2.48367000e+00],
        [2.08076811e-03, 3.10771763e-01, 6.35176957e-01, 9.41579223e-01, 1.27896261e+00,
         1.59053111e+00, 1.90248144e+00, 2.23310137e+00, 2.49199414e+00],
        [2.05414393e-03, 3.12963068e-01, 6.46444023e-01, 9.58285809e-01, 1.27928627e+00,
         1.59574628e+00, 1.89732456e+00, 2.22769356e+00, 2.50136900e+00]])

    traceErrorMNIST = torch.tensor([[0.24018253, 0.02369082, 0.01655071, 0.01396034, 0.01144823, 0.01047849,
         0.00952990, 0.00854061, 0.00797484],
        [0.23493132, 0.02441182, 0.01750353, 0.01347727, 0.01155000, 0.01005868,
         0.00924403, 0.00861205, 0.00908603],
        [0.23430178, 0.02245663, 0.01697846, 0.01301396, 0.01177753, 0.01026242,
         0.00972797, 0.00851057, 0.00796853],
        [0.23591031, 0.02403234, 0.01629571, 0.01434168, 0.01231583, 0.01095246,
         0.00969691, 0.00893384, 0.00820101],
        [0.22689532, 0.02337767, 0.01722954, 0.01429517, 0.01214339, 0.01030727,
         0.00986281, 0.00866913, 0.00863269],
        [0.23081490, 0.02363457, 0.01650987, 0.01375823, 0.01188187, 0.01073340,
         0.00933890, 0.00959375, 0.00921414],
        [0.25124684, 0.02325045, 0.01749201, 0.01358495, 0.01248278, 0.01140859,
         0.00950113, 0.00907979, 0.00792553],
        [0.23507045, 0.02369924, 0.01681624, 0.01345407, 0.01110588, 0.01080476,
         0.00938544, 0.00899983, 0.00792988],
        [0.23164381, 0.02406703, 0.01658302, 0.01367255, 0.01114006, 0.00976226,
         0.00966519, 0.00825947, 0.00789442],
        [0.23012693, 0.02313585, 0.01634406, 0.01420588, 0.01146220, 0.01042613,
         0.00921410, 0.00794939, 0.00808467],
        [0.22919241, 0.02389409, 0.01679990, 0.01348217, 0.01199206, 0.01094741,
         0.00984883, 0.00837392, 0.00826933],
        [0.23617810, 0.02385402, 0.01777303, 0.01367427, 0.01193456, 0.01086325,
         0.00972210, 0.00953787, 0.00868386],
        [0.22457570, 0.02384206, 0.01601511, 0.01331227, 0.01195829, 0.00981491,
         0.00935723, 0.00853830, 0.00810188],
        [0.22569421, 0.02354341, 0.01674058, 0.01408734, 0.01217772, 0.01048605,
         0.00958908, 0.00875280, 0.00847875],
        [0.23752135, 0.02474259, 0.01603253, 0.01287877, 0.01180999, 0.01017651,
         0.00992901, 0.00850735, 0.00869310],
        [0.23726499, 0.02293676, 0.01636312, 0.01349324, 0.01208540, 0.01093696,
         0.00915058, 0.00888785, 0.00856469],
        [0.22124900, 0.02183123, 0.01629734, 0.01310080, 0.01180154, 0.01033192,
         0.00984676, 0.00878596, 0.00819684],
        [0.23405622, 0.02429814, 0.01630274, 0.01293633, 0.01139560, 0.01045285,
         0.00971966, 0.00889976, 0.00813898],
        [0.24091603, 0.02339127, 0.01599100, 0.01363607, 0.01187595, 0.01049466,
         0.00934176, 0.00942123, 0.00768393],
        [0.23984380, 0.02154738, 0.01641268, 0.01327604, 0.01174884, 0.01027578,
         0.00978789, 0.00868630, 0.00847579]])

    exactTimingMNIST =  torch.tensor([[0.00443920], [0.00433254], [0.00421174], [0.00446614], [0.00457475], [0.00422867],
                                [0.00455546], [0.00507485], [0.00423210], [0.00486810], [0.00440400], [0.00489318],
                                [0.00428192], [0.00420458], [0.00457610], [0.00430054], [0.00442006], [0.00423517],
                                [0.00457062], [0.00422797]])


    lTimeExact = [exactTimingMiniboone, exactTimingBDS, exactTimingMNIST]

    plotTraceCompare(domainMiniboone,domainBSDS,domainMNIST,
                     approxTimingMiniboone, approxTimingBDS, approxTimingMNIST,
                     traceErrorMiniboone, traceErrorBDS, traceErrorMNIST,
                     lTimeExact, bErrBar=True)




