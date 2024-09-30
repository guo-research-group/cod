from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pickle


def rho_aligned(img, current_rho, target_rho):
    u = np.arange(0, 1920, 1)
    v = np.arange(0, 1200, 1)

    uu, vv = np.meshgrid(u, v)

    # uc-vc model
    alpha1 = -0.02003
    beta1 = 0.0002439
    gamma1 = 20.8386

    alpha2 = 0.0001309
    beta2 = -0.02032
    gamma2 = 16.39735

    u0 = (
        uu
        - beta1 * target_rho / (beta2 * target_rho + 1) * (vv - gamma2 * target_rho)
        - gamma1 * target_rho
    ) / (alpha1 * target_rho + 1)
    v0 = (vv - gamma2 * target_rho) / (beta2 * target_rho + 1)

    a = alpha1 * u0 + beta1 * v0 + gamma1
    b = alpha2 * u0 + beta2 * v0 + gamma2

    u1 = a * current_rho + u0
    v1 = b * current_rho + v0
    u1 = np.clip(u1, 0, 1918)
    v1 = np.clip(v1, 0, 1198)
    x = u1.astype(int)
    y = v1.astype(int)

    x1, x2 = x, x + 1
    y1, y2 = y, y + 1

    f1 = (x2 - u1) * img[y2, x1] + (u1 - x1) * img[y2, x2]
    f2 = (x2 - u1) * img[y1, x1] + (u1 - x1) * img[y1, x2]

    img_aligned = (v1 - y1) * f1 + (y2 - v1) * f2
    return img_aligned


def getBinningImage(image, windowRadius):
    """
    Bin image by an R x R window
    """
    if windowRadius <= 1:
        return image
    imageSum = np.zeros(image.shape)
    for i in range(windowRadius):
        for j in range(windowRadius):
            imageSum += np.roll(image, (-i, -j), axis=(0, 1))
    imageAvg = imageSum / (windowRadius**2)

    return imageAvg[::windowRadius, ::windowRadius]


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


def getDepthMap(
    imgrhoPlus,
    imgrhoMinus,
    imgSigmaPlus,
    imgSigmaMinus,
    params,
    I_rho=None,
    I_Sigma=None,
    kernelSize=5,
):
    rho = params["rho"]
    Sigma = params["Sigma"]
    delta_rho = params["Delta_rho"]
    delta_Sigma = params["Delta_Sigma"]
    sensorDistance = params["sensorDistance"]

    if I_rho is None:
        I_rho = (imgrhoPlus - imgrhoMinus) / 2
    if I_Sigma is None:
        I_Sigma = (imgSigmaPlus - imgSigmaMinus) / 2

    varAlpha = delta_Sigma * sensorDistance
    varBeta = -delta_rho * sensorDistance * Sigma
    varGamma = -delta_Sigma + sensorDistance * rho * delta_Sigma

    V = varAlpha * I_rho
    W = varBeta * I_Sigma + varGamma * I_rho

    # depth from the least square solution of a 5x5 V, W patch
    kernel = np.ones((kernelSize, 1))
    VW = signal.convolve2d(V * W, kernel, "same", "symm")
    VW = signal.convolve2d(VW, kernel.T, "same", "symm")
    W2 = signal.convolve2d(W**2, kernel, "same", "symm")
    W2 = signal.convolve2d(W2, kernel.T, "same", "symm")
    Zkf = np.divide(VW, W2, out=np.zeros_like(V), where=W != 0)

    return Zkf


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
    # You can replace this block with your dataset ================================================
    # Data Examplespy
    # Please use pickle files "Motorized_SingleScene_*.pkl"
    filepath = "./data/Motorized_SingleScene_Allmethods_Table.pkl"

    dataDicts = pickle.load(open(filepath, "rb"))
    SigmaPlusIndex = 2
    SigmaMinusIndex = 0
    SigmaIndex = 1
    rhoPlusIndex = 2
    rhoMinusIndex = 0
    rhoIndex = 1

    # Get optical power
    rhoMinus = dataDicts[0][SigmaIndex][rhoMinusIndex]["OP"]
    rhoPlus = dataDicts[0][SigmaIndex][rhoPlusIndex]["OP"]
    rhoLens = dataDicts[0][SigmaIndex][rhoIndex]["OP"]
    # Get aperture radius
    SigmaPlus = dataDicts[0][SigmaPlusIndex][rhoIndex]["Sigma"]
    SigmaMinus = dataDicts[0][SigmaMinusIndex][rhoIndex]["Sigma"]
    Sigma = dataDicts[0][SigmaIndex][rhoIndex]["Sigma"]

    # Get images
    imgSigmaPlus = dataDicts[0][SigmaPlusIndex][rhoIndex]["Img"].astype(np.float64)
    imgSigmaMinus = dataDicts[0][SigmaMinusIndex][rhoIndex]["Img"].astype(np.float64)
    imgrhoPlus = dataDicts[0][SigmaIndex][rhoPlusIndex]["Img"].astype(np.float64)
    imgrhoMinus = dataDicts[0][SigmaIndex][rhoMinusIndex]["Img"].astype(np.float64)

    params = {
        "rho": 8.9 + rhoLens,       # Optical Power
        "Sigma": 0.0025,            # Aperture Radius
        "Delta_rho": 0.06,          # (rhoPlus - rhoMinus) / 2
        "Delta_Sigma": 0.0010,      # (SigmaPlus - SigmaMinus) / 2
        "sensorDistance": 0.1100,   # Flange focal distance
    }

    # bin 4*4 pixels
    binning_windowSize = 4

    # =============================================================================================

    # Brightness Alignment
    imgSigmaPlus = imgSigmaPlus * ((Sigma / SigmaPlus) ** 2)
    imgSigmaMinus = imgSigmaMinus * ((Sigma / SigmaMinus) ** 2)

    # rho Alignment
    imgrhoPlus = rho_aligned(imgrhoPlus, rhoPlus, rhoLens)
    imgrhoMinus = rho_aligned(imgrhoMinus, rhoMinus, rhoLens)

    # bin images
    imgSigmaPlus = getBinningImage(imgSigmaPlus, binning_windowSize)
    imgSigmaMinus = getBinningImage(imgSigmaMinus, binning_windowSize)
    imgrhoPlus = getBinningImage(imgrhoPlus, binning_windowSize)
    imgrhoMinus = getBinningImage(imgrhoMinus, binning_windowSize)

    # Remove bias
    imgSigmaPlus = removeBiasISigma(imgSigmaPlus)
    imgSigmaMinus = removeBiasISigma(imgSigmaMinus)
    imgrhoPlus = removeBiasISigma(imgrhoPlus)
    imgrhoMinus = removeBiasISigma(imgrhoMinus)

    # Calculate the Depth Map
    ZMap = getDepthMap(imgrhoMinus, imgrhoPlus, imgSigmaMinus, imgSigmaPlus, params)

    # Show the raw depth map
    fig = plt.figure(figsize=(12, 12), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    plot_ZMap = ax.imshow(
        ZMap,
        vmin=0.4,
        vmax=1.4,
        cmap="rainbow",
    )
    fig.colorbar(plot_ZMap, ax=ax, fraction=0.046, pad=0.04)
    plt.show()

    # obtain confidence map
    confidenceMap = (imgrhoPlus - imgrhoMinus) ** 2
    # confidence sparsity in [0, 1)
    sparsity = 0.5
    filteredZMap = filterResultByConfidenceSparsity(ZMap, confidenceMap, sparsity)

    # Show the filtered depth map
    fig = plt.figure(figsize=(12, 12), dpi=100)
    ax = fig.add_subplot(1, 1, 1)
    plot_ZMap = ax.imshow(
        filteredZMap,
        vmin=0.4,
        vmax=1.4,
        cmap="rainbow",
    )
    fig.colorbar(plot_ZMap, ax=ax, fraction=0.046, pad=0.04)
    plt.show()

    return


if __name__ == "__main__":
    main()
