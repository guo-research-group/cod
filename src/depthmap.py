from scipy import signal
import matplotlib.pyplot as plt
import numpy as np


def getDepthMap(I_rho, I_A, params, kernelSize=5):
    # optical power, denoted as rho
    rho = params["rho"]
    apertureRadius = params["apertureRadius"]
    sensorDistance = params["sensorDistance"]

    varAlpha = 1
    varBeta = rho - 1 / sensorDistance
    varGamma = -apertureRadius

    V = varAlpha * np.ones(I_rho.shape)
    W = varBeta + varGamma * np.divide(
        I_A, I_rho, out=np.zeros_like(I_A), where=I_rho != 0
    )

    # depth from the least square solution of a 5x5 V, W patch
    # can be replaced by for loop, with lower FLOPOP
    if kernelSize > 1:
        uniformFilter = np.ones((kernelSize, 1))
        VW = V * W
        square_W = W**2
        VW = signal.convolve2d(VW, uniformFilter, "same", "symm")
        VW = signal.convolve2d(VW, uniformFilter.T, "same", "symm")
        square_W = signal.convolve2d(square_W, uniformFilter, "same", "symm")
        square_W = signal.convolve2d(square_W, uniformFilter.T, "same", "symm")
        ZMap = np.divide(VW, square_W, out=np.zeros_like(VW), where=square_W != 0)
    else:
        ZMap = np.divide(V, W, out=np.zeros_like(V), where=W != 0)

    return ZMap


def filterResultByConfidenceSparsity(ZArray, ZConfidence, confidence_level=0.95):
    if confidence_level == 0:
        return np.array(ZArray)

    ZConfidence_ = ZConfidence.flatten()
    ZConfidence_ = ZConfidence_[ZConfidence_ < np.inf]
    ZConfidence_f = np.where(ZConfidence < np.inf, ZConfidence, np.nan)
    sortZkfConfidence = np.sort(ZConfidence_)
    confidenceLevel = sortZkfConfidence[
        int((len(sortZkfConfidence) - 1) * confidence_level)
    ]

    return np.where(ZConfidence_f > confidenceLevel, ZArray, np.nan)


def main():
    """
    Example Code
    """
    # params address
    params_path = "./params.npy"
    # I_rho address
    I_rho_path = "./I_rho.npy"
    # I_A address
    I_A_path = "./I_A.npy"

    # Data Examples
    # params = {
    #     "rho": 10.1,
    #     "apertureRadius": 0.0025,
    #     "sensorDistance": 0.1100,
    #     "kernelSize": 5,
    # }
    # I_rho = (I_rho_1 - I_rho_2) / 2
    # I_A = (I_A_1 - I_A_2) / 2
    params = np.load(params_path)
    I_rho = np.load(I_rho_path)
    I_A = np.load(I_A_path)

    # Calculate the Depth Map
    ZMap = getDepthMap(I_rho, I_A, params, params["kernelSize"])

    # Show the raw depth map
    fig = plt.figure(figsize=(12, 12), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    plot_ZMap = ax.imshow(ZMap)
    fig.colorbar(plot_ZMap, ax=ax, fraction=0.046, pad=0.04)

    confidenceMap = I_rho**2
    # confidence sparsity in [0, 1)
    sparsity = 0.5
    filteredZMap = filterResultByConfidenceSparsity(ZMap, confidenceMap, sparsity)

    # Show the filtered depth map
    fig = plt.figure(figsize=(12, 12), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    plot_ZMap = ax.imshow(filteredZMap)
    fig.colorbar(plot_ZMap, ax=ax, fraction=0.046, pad=0.04)

    return


if __name__ == "__main__":
    main()
