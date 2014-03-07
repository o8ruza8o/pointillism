from colorCluster import *
import pylab as pl
import random
import numpy as np
from scipy import *
from scipy import misc, interpolate, spatial, random, ndimage

class Point:
    def __init__(self, position, color):
        self.p = position
        self.c = color

def to_grayscale(img, vector = np.ones(3, dtype = float), power = 1.0):
    "If img is a color image (3D array), convert it to  a2D array."
    if len(img.shape) == 3:
        img2d = ((np.abs(img - vector))**power).sum(axis = -1)
        np.set_printoptions(precision=3)
    else:
        img2d = img

    return (img2d.max() - img2d)/(img2d.max()-img2d.min()) + np.finfo(np.float).eps


def plotit(centroids, cvec = np.ones(3, dtype = float)):
    ctuple = tuple(cvec)
    pl.plot(centroids[:][1], -centroids[:][0], lw=0, marker='o', markersize=2, color=ctuple, alpha=1.0, markeredgecolor='none') # for '.' use None instead of 'none'
    pl.axis('image')

def nPointSeed(image, n):
    "Seed according to the distribution in the image."
    # Compute a CDF function across the flattened image
    imageCDF = image.flatten().cumsum()
    imageCDF /= 1.0 * imageCDF.max()
    imageCDF += np.finfo(np.float).eps

    # Function to turn a random point in the CDF into a random index in the image
    indexInterpolator = interpolate.interp1d(imageCDF, arange(imageCDF.size))

    # Set to collect the UNIQUE indices
    indexContainer = set()
    while len(indexContainer) < n:
        # Generate at most the number of points remaining
        maxToGenerate = n - len(indexContainer)
        randomCDFValues = random.uniform(0, 1.0, maxToGenerate)

        # Back them into indices
        iInterp = indexInterpolator(randomCDFValues)
        iInterp = np.round(iInterp).astype(uint32)

        # Add them to the set
        indexContainer.update(iInterp)

    # Break them out of the set
    iInterp = array(list(indexContainer))

    # Compute the equivalent xy
    xCoords = (iInterp // image.shape[1]).astype(int32)
    yCoords = (iInterp %  image.shape[1]).astype(int32)

    # Return them glued together.
    return np.c_[xCoords, yCoords].T

def optimize(generators, centroids, n, shape):
    count = 0
    while True:
        gens = np.ones(shape, dtype = int)
        gen2region = np.zeros(shape, dtype = int)
        for i in range(n):
            gens[generators[0, i], generators[1, i]] = 0
            gen2region[generators[0, i], generators[1, i]] = i
            
        # divede the image into regions
        regionXYs = ndimage.distance_transform_edt(gens, return_distances = False, return_indices = True)
        region = np.zeros(shape, dtype = int)
            
        for (i, j), junk in np.ndenumerate(region):
            r = regionXYs[:, i, j]
            region[i, j] = gen2region[r[0], r[1]]
                    
        # calculate the centroid of each region
        centroids = np.array(ndimage.measurements.center_of_mass(gray_img, region, range(n))).T
                    
        # move each region's generator to it's centroid
        if (np.abs(generators - centroids)).max() < 5.0:
            break

        generators[:] = np.round(centroids.copy())
        count += 1
    return centroids


if __name__=='__main__':
    # get the image to be stippled
    filename = "shoe.png"
    img = misc.imread(filename)
    points = []
    K = 1

    if (len(img.shape) == 3 and (K > 1)):
        # if color image cluster it to K colors 
        carray, frequency = colorCluster(img, 8)
    else:  # black stipples only
        K = 1
        carray = np.array([[0.0] * 3])
        frequency = [mean(img)/255.0 * img.shape[0] * img.shape[1]]
        
    for i in range(K):
        # project the image along a color vector
        col = carray[i]
        gray_img = to_grayscale(img.astype(np.float), col).squeeze()

        # get the dimensions of image and number of points to replace it with
        imshape = gray_img.shape
        nPoints = int(float(frequency[i]) / 50.0)

        # gimme some numbers
        np.set_printoptions(precision=3)
        print "{0:.3f}".format(mean(gray_img)), col, nPoints

        # initialize generators and centroids arrays
        generators =  np.zeros([2, nPoints], dtype = np.int)
        centroids = np.zeros([2, nPoints], dtype = np.float)

        # seed generators and optimize centroids
        generators = nPointSeed(gray_img, nPoints)
        centroids = optimize(generators, centroids, nPoints, imshape)

        # append the result to the list of points:
        for cen in centroids.T:
            points.append(Point(cen, col))

    # randomise order of points to plot
    print len(points), " points polotted!"
    random.shuffle(points)
    pl.figure()
    for point in points:
        plotit(point.p, point.c/255.0)
    pl.axis('image')
    frame1 = pl.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    pl.savefig(filename.split(".")[0]+'-point.' + filename.split(".")[-1], 
               transparent=True, bbox_inches='tight', 
               pad_inches=0.0, frameon=None)

    pl.figure()
    pl.imshow(img)
    pl.axis('image')
    if K == 1: 
        pl.gray()
    frame2 = pl.gca()
    frame2.axes.get_xaxis().set_visible(False)
    frame2.axes.get_yaxis().set_visible(False)
    pl.savefig(filename.split(".")[0]+'-plot.' + filename.split(".")[-1], 
               transparent=True, bbox_inches='tight', 
               pad_inches=0.0, frameon=None)
    pl.show()
