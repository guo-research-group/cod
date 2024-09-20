import numpy as np
import scipy
import pickle
from matplotlib import pyplot as plt
import matplotlib
import glob
import pdb
import cv2
import os
from scipy import signal


WORKING_RANGE = np.linspace(0.45, 1.41, 97)
HEATMAP_RANGE = [
    [WORKING_RANGE.min(), WORKING_RANGE.max()],
    [WORKING_RANGE.min(), WORKING_RANGE.max()],
]


def getImageWindow(img: np.ndarray, position: np.ndarray):
    """
    Crop images
    """
    return img[
        position[0, 0] : position[0, 1],
        position[1, 0] : position[1, 1],
    ]


def getBinningImagesFast(image, windowRadius):
    """
    Bin image by an R x R window
    """
    kernel = np.ones((windowRadius, 1)) / windowRadius
    image = signal.convolve2d(image, kernel, "same", "symm")[::windowRadius, :]
    image = signal.convolve2d(image, kernel.T, "same", "symm")[:, ::windowRadius]

    return image


def removeBiasISigma(image):
    """
    (Optional)
    Remove the internal reflection
    It depends on whether there is similar aberration
    """
    window = 21
    kernel = np.ones((window, 1)) / window
    image_blurred = signal.convolve2d(image, kernel, "same", "symm")
    image_blurred = signal.convolve2d(image_blurred, kernel.T, "same", "symm")

    return image - image_blurred


def plotSingleResult(Zkf, Ztrue, pathname, title=None):

    fig = plt.figure(figsize=(5, 5), dpi=100)

    Zkfplot_N = fig.add_subplot(1, 1, 1)
    heatmap, xedges, yedges = np.histogram2d(
        Ztrue.flatten(), Zkf.flatten(), bins=97, range=HEATMAP_RANGE
    )
    heatmap = heatmap.T
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    # vmax = np.percentile(heatmap.flatten(), 99)
    ZkfHist = Zkfplot_N.imshow(
        heatmap,
        extent=extent,
        origin="lower",
        # , vmax=vmax, vmin=0
    )
    fig.colorbar(ZkfHist, ax=Zkfplot_N, fraction=0.046, pad=0.04)
    Zkfplot_N.plot(WORKING_RANGE, WORKING_RANGE, c="white")
    Zkfplot_N.set_xlabel("True Depth (m)")
    Zkfplot_N.set_ylabel("Estimated Depth (m)")
    if title is not None:
        Zkfplot_N.set_title(title)

    fig.tight_layout()
    plt.savefig(pathname)
    plt.close(fig)

    return heatmap


def getDepthMap(
    IrhoPlus,
    IrhoMinus,
    IAPlus,
    IAMinus,
    params,
    kernelsize=5,
):
    """
    Calculate the depth map
    """
    rho = params["rho"]
    Sigma = params["Sigma"]
    delta_rho = params["delta_rho"]
    delta_Sigma = params["delta_Sigma"]
    sensorDistance = params["sensorDistance"]
    photon_per_brightness_level = params["photon_per_brightness_level"]

    Irho = (IrhoPlus - IrhoMinus) / 2
    IA = (IAPlus - IAMinus) / 2

    varAlpha = delta_Sigma * sensorDistance
    varBeta = -delta_rho * sensorDistance * Sigma
    varGamma = -delta_Sigma + sensorDistance * rho * delta_Sigma

    V = varAlpha * Irho
    W = varBeta * IA + varGamma * Irho

    kernel = np.ones((kernelsize, 1))

    if len(V.shape) == 3:
        VW = np.sum(V * W, axis=-1)
        W2 = np.sum(W**2, axis=-1)
        IA2 = np.sum(IA * IA, axis=-1)
        Irho2 = np.sum(Irho * Irho, axis=-1)
        IAIrho = np.sum(IA * Irho, axis=-1)
    else:
        VW = V * W
        W2 = W**2
        IA2 = IA * IA
        Irho2 = Irho * Irho
        IAIrho = IA * Irho

    VW = signal.convolve2d(VW, kernel, "same", "symm")
    VW = signal.convolve2d(VW, kernel.T, "same", "symm")
    W2 = signal.convolve2d(W2, kernel, "same", "symm")
    W2 = signal.convolve2d(W2, kernel.T, "same", "symm")

    Z = np.divide(VW, W2, out=np.zeros_like(VW), where=W2 != 0)

    IA2 = signal.convolve2d(IA2, kernel, "same", "symm")
    IA2 = signal.convolve2d(IA2, kernel.T, "same", "symm")

    Irho2 = signal.convolve2d(Irho2, kernel, "same", "symm")
    Irho2 = signal.convolve2d(Irho2, kernel.T, "same", "symm")

    IAIrho = signal.convolve2d(IAIrho, kernel, "same", "symm")
    IAIrho = signal.convolve2d(IAIrho, kernel.T, "same", "symm")

    LSratios = IAIrho / Irho2

    # ZConfidence = Irho2 + IA2
    ZConfidence = Irho2

    return Z, ZConfidence, LSratios


def filterResultByConfidenceSparsity(
    ZArray,
    ZConfidence,
    confidence_level=0.95,
    working_range=WORKING_RANGE,
    outliersFiltering=False,
):
    """
    Filter out the result with low confidence value
    """
    if confidence_level == 0:
        if outliersFiltering is True:
            return np.where(
                (ZArray < working_range.max()) & (ZArray > working_range.min()),
                ZArray,
                np.nan,
            )
        else:
            return ZArray

    ZConfidence_ = ZConfidence.flatten()
    ZConfidence_ = ZConfidence_[ZConfidence_ < np.inf]
    ZConfidence_ = ZConfidence_[ZConfidence_ != np.nan]
    ZConfidence_f = np.where(ZConfidence < np.inf, ZConfidence, np.nan)
    sortZkfConfidence = np.sort(ZConfidence_)
    confidenceLevel = sortZkfConfidence[
        int((len(sortZkfConfidence) - 1) * confidence_level)
    ]

    if outliersFiltering is True:
        ZArray_ = np.where(
            (ZArray < working_range.max()) & (ZArray > working_range.min()),
            ZArray,
            np.nan,
        )
    else:
        ZArray_ = ZArray

    return np.where(ZConfidence_f > confidenceLevel, ZArray_, np.nan)


def createLUT(ratios, Ztrue, display_LUT=False):
    """
    Create a Lookup Table (LUT) based on the ratio IA/Irho
    """
    Nbins = 500
    vmin = np.percentile(ratios, 1)
    vmax = np.percentile(ratios, 98)

    ratio_bins = np.linspace(vmin, vmax, Nbins)
    idx = np.digitize(ratios, ratio_bins)
    Z_median = []
    for i in range(len(ratio_bins)):
        Z_median.append(np.median(Ztrue[idx == i]))

    LUT = np.stack([ratio_bins, Z_median], -1)

    if display_LUT:
        plt.plot(ratio_bins, Z_median, ".")
        plt.xlabel("$I_A/I_\\rho$")
        plt.ylabel("Mapped Depth (m)")
        plt.show()

    return LUT


def applyLUT(ratios, LUT):
    """
    Apply LUT to a vector
    """
    ratio_bins = LUT[..., 0]
    Z = LUT[..., 1]
    idx = np.digitize(ratios, ratio_bins)
    idx[idx == len(ratio_bins)] = len(ratio_bins) - 1
    Z_pred = Z[idx]

    return Z_pred


def getHeatmapByLUT(params, filepath, indexes):
    SigmaPlusIndex = 2
    SigmaMinusIndex = 0
    SigmaIndex = 1
    rhoPlusIndex = indexes[2]
    rhoMinusIndex = indexes[0]
    rhoIndex = indexes[1]

    ksize = 1
    ksizes = [3, 5, 7, 9, 11, 13, 15, 17]
    ktype = "None"
    binning_windowSize = 1
    confidenceLevels = [0, 0.5, 0.9, 0.95, 0.99]
    confidenceTypes = ["C2"]
    xKernel = np.array([[-0.5, 0, 0.5]])
    yKernel = np.array([[-0.5], [0], [0.5]])

    dataDicts = pickle.load(open(filepath, "rb"))

    rhoMinus = dataDicts[0][SigmaIndex][rhoMinusIndex]["OP"]
    rhoPlus = dataDicts[0][SigmaIndex][rhoPlusIndex]["OP"]
    rhoLens = dataDicts[0][SigmaIndex][rhoIndex]["OP"]
    SigmaPlus = dataDicts[0][SigmaPlusIndex][rhoIndex]["Sigma"]
    SigmaMinus = dataDicts[0][SigmaMinusIndex][rhoIndex]["Sigma"]
    Sigma = dataDicts[0][SigmaIndex][rhoIndex]["Sigma"]

    params["rho"] = params["rho0"] + rhoLens

    print(
        "rho plus:%f\nrho minus:%f\nrho:%f\nSigma plus:%f\nSigma minus:%f\nSigma:%f"
        % (rhoPlus, rhoMinus, rhoLens, SigmaPlus, SigmaMinus, Sigma)
    )
    print(params)

    filename = os.path.basename(filepath).split(".")[0]
    if indexes == [0, 1, 2]:
        filename = filename + "1"
    else:
        filename = filename + "2"
    print("Current file:", filename)

    for binning_windowSize in [4]:
        outputPath = "./" + filename + "_LUT_binning_windowSize%d" % binning_windowSize
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        outputPath += "/"

        texture_position = (
            np.array([[500, 900], [800, 1200]], dtype=np.int64) / binning_windowSize
        ).astype(np.int64)

        if not os.path.exists(outputPath + "savedData.pkl"):
            savedData = []
            for i in range(len(dataDicts)):
                loc = dataDicts[i][SigmaIndex][rhoIndex]["Loc"]

                rho = dataDicts[i][SigmaIndex][rhoIndex]["OP"]
                Sigma = dataDicts[i][SigmaIndex][rhoIndex]["Sigma"]
                rhos = np.array([x["OP"] for x in dataDicts[i][SigmaIndex]])
                images = np.array([x["Img"] for x in dataDicts[i][SigmaIndex]]).astype(
                    np.float64
                )

                SigmaPlus = dataDicts[i][SigmaPlusIndex][rhoIndex]["Sigma"]
                SigmaMinus = dataDicts[i][SigmaMinusIndex][rhoIndex]["Sigma"]
                imgSigmaPlus = dataDicts[i][SigmaPlusIndex][rhoIndex]["Img"].astype(
                    np.float64
                )
                imgSigmaMinus = dataDicts[i][SigmaMinusIndex][rhoIndex]["Img"].astype(
                    np.float64
                )

                imgSigmaPlus = getImageWindow(
                    getBinningImagesFast(imgSigmaPlus, binning_windowSize),
                    texture_position,
                )
                imgSigmaMinus = getImageWindow(
                    getBinningImagesFast(imgSigmaMinus, binning_windowSize),
                    texture_position,
                )

                # Brightness Alignment
                imgSigmaPlus = imgSigmaPlus * ((Sigma / SigmaPlus) ** 2)
                imgSigmaMinus = imgSigmaMinus * ((Sigma / SigmaMinus) ** 2)

                imgrhoPlus = getImageWindow(
                    getBinningImagesFast(images[rhoPlusIndex], binning_windowSize),
                    texture_position,
                )
                imgrhoMinus = getImageWindow(
                    getBinningImagesFast(images[rhoMinusIndex], binning_windowSize),
                    texture_position,
                )

                imgSigma = removeBiasISigma(imgSigmaPlus - imgSigmaMinus)
                imgrho = removeBiasISigma(imgrhoPlus - imgrhoMinus)

                ZMap, CMap, LSratio = getDepthMap(
                    imgrho,
                    0,
                    imgSigma,
                    0,
                    params,
                )

                fig = plt.figure(figsize=(20, 20), dpi=100)
                ax = fig.add_subplot(1, 1, 1)
                plot = ax.imshow(
                    getImageWindow(
                        getBinningImagesFast(imgSigmaPlus, binning_windowSize),
                        texture_position,
                    )
                )
                ax.set_title("Texture Area, Location:%.0f" % loc)
                fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
                fig.savefig(outputPath + "OutputFigure%d.png" % i)
                plt.close(fig)

                TempParams = params.copy()
                TempParams["depth"] = 0.93 + loc * 4e-7

                savedData.append(
                    [
                        loc,
                        imgrho,
                        imgSigma,
                        ZMap,
                        CMap,
                        LSratio,
                        TempParams,
                    ]
                )
            pickle.dump(savedData, open(outputPath + "savedData.pkl", "wb"))

        savedData = pickle.load(open(outputPath + "savedData.pkl", "rb"))

        ratios = []
        Z_true = []
        conf = []
        Z_cal = []
        for i in range(len(savedData)):
            (
                loc,
                imgrho,
                imgSigma,
                ZMap,
                CMap,
                LSratio,
                TempParams,
            ) = savedData[i]
            conf.append(CMap)
            ratios.append(LSratio)
            Z_true.append(np.ones(LSratio.shape) * TempParams["depth"])
            Z_cal.append(ZMap)
        # create LUT
        ratios = np.stack(ratios).flatten()
        Z_true = np.stack(Z_true).flatten()
        conf = np.stack(conf).flatten()
        Z_cal = np.stack(Z_cal).flatten()

        LUT = createLUT(ratios, Z_true, display_LUT=False)

        # # Display LUT
        # plt.figure(figsize=(5, 5),dpi=100)
        # plt.plot(LUT[:, 0], LUT[:, 1], ".")
        # plt.plot(LUT[:, 0], Z_theory(LUT[:, 0], params))
        # plt.xlabel("$r$")
        # plt.ylabel("Depth (m)")
        # plt.show()

        for confLevel in confidenceLevels:
            filterIdx = filterResultByConfidenceSparsity(
                np.ones_like(conf), conf, confLevel
            )
            ratios_filtered = ratios[~np.isnan(filterIdx)]
            Z_true_filtered = Z_true[~np.isnan(filterIdx)]
            Z_cal_filtered = Z_cal[~np.isnan(filterIdx)]

            # apply LUT
            Z_pred_filtered = applyLUT(ratios_filtered, LUT)

            cal_heatmap, vmin, vmax = plotSingleResult(
                Z_cal_filtered,
                Z_true_filtered,
                outputPath + f"Calculation_{confLevel:.2f}.png",
                None,
            )

            plotSingleResult(
                Z_pred_filtered,
                Z_true_filtered,
                outputPath + f"LUT_{confLevel:.2f}.png",
                None,
                vmin,
                vmax,
            )

            # np.save(outputPath + f"LUT_{confLevel:.2f}", LUT_heatmap)

            # Filter by depth
            Z_cal_filtered_per_depth = []
            Z_true_filtered_per_depth = []
            for i in range(len(savedData)):
                (
                    loc,
                    imgrho,
                    imgSigma,
                    ZMap,
                    CMap,
                    LSratio,
                    TempParams,
                ) = savedData[i]
                Z_cal_filtered_per_depth.append(
                    filterResultByConfidenceSparsity(ZMap, CMap, confLevel)
                )
                Z_true_filtered_per_depth.append(
                    np.ones(Z_cal_filtered_per_depth[-1].shape) * TempParams["depth"]
                )

            Z_cal_filtered_per_depth = np.stack(Z_cal_filtered_per_depth).flatten()
            Z_true_filtered_per_depth = np.stack(Z_true_filtered_per_depth).flatten()

            plotSingleResult(
                Z_cal_filtered_per_depth,
                Z_true_filtered_per_depth,
                outputPath + f"Per_depth_filtering_{confLevel:.2f}.png",
                None,
                vmin,
                vmax,
            )

    # isExist = os.path.isfile("./heatmap.pkl")
    # if isExist:
    #     file = open("./heatmap.pkl", "rb")
    #     heatmpDicts = pickle.load(file)
    #     file.close()
    # else:
    #     # if the file does not exist, creat one
    #     heatmpDicts = []

    # heatmpDicts.append(savedData)
    # file = open("./heatmap.pkl", "wb")
    # pickle.dump(heatmpDicts, file)
    # file.close()

    return
