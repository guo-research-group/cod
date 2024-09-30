import numpy as np
import scipy
import pickle
from matplotlib import pyplot as plt
import matplotlib
import glob
import pdb
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


def plotSingleResult(Zkf, Ztrue, pathname, title=None, vmax=None, vmin=None):

    fig = plt.figure(figsize=(5, 5), dpi=100)

    Zkfplot_N = fig.add_subplot(1, 1, 1)
    Zkfplot_N.plot(WORKING_RANGE, WORKING_RANGE, c="white")
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
    if vmax is not None and vmin is not None:
        ZkfHist.set_clim(vmin=vmin, vmax=vmax)
    fig.colorbar(ZkfHist, ax=Zkfplot_N, fraction=0.046, pad=0.04)
    Zkfplot_N.set_xlabel("True Depth (m)")
    Zkfplot_N.set_ylabel("Estimated Depth (m)")

    fig.tight_layout()
    plt.savefig(pathname)
    plt.close(fig)

    vmax = np.max(heatmap)
    vmin = np.min(heatmap)

    return heatmap, vmax, vmin


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


def getMAEByLUT(params, filepath, indexes, botIndex=None, upIndex=None):
    SigmaPlusIndex = 2
    SigmaMinusIndex = 0
    SigmaIndex = 1
    rhoPlusIndex = indexes[2]
    rhoMinusIndex = indexes[0]
    rhoIndex = indexes[1]

    binning_windowSize = 1
    confidenceLevels = [0, 0.5, 0.9, 0.95, 0.99]

    dataDicts = pickle.load(open(filepath, "rb"))
    dataDicts_extra = pickle.load(
        open("./data/Motorized_LinearSlide_Texture5_NewParameter_Extend3.pkl", "rb")
    )
    dataDicts = dataDicts + dataDicts_extra

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
        outputPath = (
            "./MAE_COD_LUT_" + filename + "_binning_windowSize%d" % binning_windowSize
        )
        if not os.path.exists(outputPath):
            os.makedirs(outputPath)
        outputPath += "/"

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

                imgSigmaPlus = getBinningImagesFast(imgSigmaPlus, binning_windowSize)
                imgSigmaMinus = getBinningImagesFast(imgSigmaMinus, binning_windowSize)

                # Brightness Alignment
                imgSigmaPlus = imgSigmaPlus * ((Sigma / SigmaPlus) ** 2)
                imgSigmaMinus = imgSigmaMinus * ((Sigma / SigmaMinus) ** 2)

                imgrhoPlus = getBinningImagesFast(
                    images[rhoPlusIndex], binning_windowSize
                )
                imgrhoMinus = getBinningImagesFast(
                    images[rhoMinusIndex], binning_windowSize
                )

                imgSigma = removeBiasISigma(imgSigmaPlus - imgSigmaMinus)
                imgrho = removeBiasISigma(imgrhoPlus - imgrhoMinus)

                fig = plt.figure(figsize=(20, 20), dpi=100)
                ax = fig.add_subplot(1, 1, 1)
                plot = ax.imshow(imgSigmaPlus)
                ax.set_title("Texture Area, Location:%.0f" % loc)
                fig.colorbar(plot, ax=ax, fraction=0.046, pad=0.04)
                fig.savefig(outputPath + "OutputFigure%d.png" % i)
                plt.close(fig)

                ZMap, CMap, LSratio = getDepthMap(
                    imgrho,
                    0,
                    imgSigma,
                    0,
                    params,
                )

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
        if upIndex is None or botIndex is None:
            upIndex = len(savedData)
            botIndex = 0

        for i in range(botIndex, upIndex):
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

        filtered_ratios = filterResultByConfidenceSparsity(ratios, conf, 0)
        LUT = createLUT(
            filtered_ratios[~np.isnan(filtered_ratios)],
            Z_true[~np.isnan(filtered_ratios)],
            display_LUT=False,
        )

        errors = []
        for confidenceLevel in confidenceLevels:
            filterIdx = filterResultByConfidenceSparsity(
                np.ones_like(conf), conf, confidenceLevel
            )

            Z_pred = applyLUT(ratios, LUT)
            Z_pred_filtered = Z_pred.copy()
            Z_pred_filtered[np.isnan(filterIdx)] = np.nan
            Z_cal_filtered = Z_cal.copy()
            Z_cal_filtered[np.isnan(filterIdx)] = np.nan

            Z_error = Z_pred_filtered

            current_error = np.nanmean(
                np.abs(
                    np.array(Z_error).reshape(upIndex - botIndex, -1)
                    - np.array(Z_true).reshape(upIndex - botIndex, -1)
                ),
                axis=-1,
            )
            current_meanDepth = np.nanmean(
                np.array(Z_error).reshape(upIndex - botIndex, -1), axis=-1
            )
            current_meanDifference = np.nanmean(
                np.abs(
                    np.array(Z_error).reshape(upIndex - botIndex, -1)
                    - np.repeat(current_meanDepth, 10000).reshape(
                        upIndex - botIndex, -1
                    )
                ),
                axis=-1,
            )
            errors.append(
                [
                    confidenceLevel,
                    current_error,
                    current_meanDepth,
                    current_meanDifference,
                ]
            )

            plotSingleResult(
                np.array(Z_pred_filtered),
                np.array(Z_true),
                outputPath + "LUT%.2f.png" % (confidenceLevel),
            )

            plotSingleResult(
                np.array(Z_cal_filtered),
                np.array(Z_true),
                outputPath + "hist%.2f.png" % (confidenceLevel),
            )

    return errors


def plotWorkingArea(errors, working_range=WORKING_RANGE):
    font = {
        "size": 32,
    }
    matplotlib.rc("font", **font)

    fig = plt.figure(figsize=(12, 12), dpi=100)
    ax = fig.add_subplot(1, 1, 1)

    plot_range = np.linspace(0.45, 1.81, 100)

    ax.set_xlim(0.45, 1.81)
    ax.set_ylim(0.45, 1.81)

    ax.plot(
        plot_range,
        plot_range,
        linewidth=1,
        color="black",
    )

    ax.plot(
        plot_range,
        plot_range * 1.1,
        linewidth=1,
        color="black",
        linestyle="dashed",
    )

    ax.plot(
        plot_range,
        plot_range * 0.9,
        linewidth=1,
        color="black",
        linestyle="dashed",
    )

    colors = ["red", "green", "blue"]

    for i in range(3):
        [
            confidenceLevel,
            current_error,
            current_meanDepth,
            current_meanDifference,
        ] = errors[i]

        sort_index = np.argsort(working_range)
        sorted_working_range = working_range[sort_index]
        sorted_current_meanDepth = current_meanDepth[sort_index]
        sorted_meanDifference = current_meanDifference[sort_index]

        ax.plot(
            sorted_working_range, sorted_current_meanDepth, linewidth=1, color=colors[i]
        )

        ax.fill_between(
            sorted_working_range,
            (sorted_current_meanDepth + sorted_meanDifference),
            (sorted_current_meanDepth - sorted_meanDifference),
            color=colors[i],
            alpha=0.2,
        )

    ax.set_xlabel("Depth (m)")
    ax.set_ylabel("Predicted Depth (m)")

    ax.grid()
    # ax.legend()
    fig.tight_layout()
    plt.show()

    return


def main():
    # Example code
    # This code is based on "depthmap.py"
    # You can write your own heatmap code with the dataset
    params = {
        "rho0": 8.9,
        "Sigma": 0.0025,
        "delta_rho": 0.06,
        "delta_Sigma": 0.0010,
        "sensorDistance": 0.1100,
        "photon_per_brightness_level": 120,
    }

    errors = getMAEByLUT(
        params, "./data/Motorized_LinearSlide_Texture5_NewParameter.pkl", [0, 1, 2]
    )

    with open("./errors_COD_extra.pkl", "wb") as f:
        pickle.dump(errors, f)
        f.close()
    with open("./errors_COD_extra.pkl", "rb") as f:
        errors_COD_extra = pickle.load(f)
        f.close()
    new_working_range = np.concatenate(
        [WORKING_RANGE, np.linspace(0.9072, 1.8672, 97)[0:98]], 0
    )
    plotWorkingArea(errors_COD_extra, working_range=new_working_range)

    return


if __name__ == "__main__":
    main()
