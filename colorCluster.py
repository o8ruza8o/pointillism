from numpy import *
from scipy import misc, ndimage
from sklearn.cluster import KMeans
from pylab import *

def colorCluster(colorImage, nColorClusters, showSegmentation = False):
    # Compute the pix image shape
    imageShape = colorImage.shape[0:2]

    # Unroll the image into RGB vector (3 x pix count)
    rgbs = colorImage.reshape((-1, 3)).astype(float64)

    # Fit it with a kmeans estimator
    kme = KMeans(init='k-means++', n_clusters=nColorClusters, n_init=10)
    kme.fit(rgbs)

    # Show the quantized image if desired
    if showSegmentation:
        result = kme.predict(rgbs).reshape(imageShape)

        projected = empty_like(colorImage)
        for i, c in enumerate("RGB"):
            projected[:,:,i] = kme.cluster_centers_[:, i].take(result)

        figure()
        imshow(colorImage)

        figure()
        imshow(projected)
        show()

    # What a derpy attr name. . .
    return kme.cluster_centers_

if __name__ == "__main__":
    cdata = ndimage.imread("color-wheel-300.png", mode="RGB")
    colorCluster(cdata, 4, showSegmentation=True)
    colorCluster(cdata, 7, showSegmentation=True)
