from colorCluster import *
import pylab as pl
import cairo
import itertools
import numpy as np
from scipy import *
from scipy import misc, interpolate, spatial, random, ndimage
from random import shuffle, randrange
import copy

class Point:
    def __init__(self, position, color):
        self.p = position
        self.c = color

def projectImage(img, vector = np.ones(3, dtype = np.float32), power = 1.0):
    "If img is a color image (3D array), convert it to a 2D array."
    if len(img.shape) == 3:
        img2d = ((np.abs(img - vector))**power).sum(axis = -1)
    else:
        img2d = img

    return (img2d.max() - img2d)/(img2d.max()-img2d.min()) + np.finfo(np.float32).eps

def drawCircle(ctx, center, radius, color):
    "Draw a circle using Cairo vector graphics."
    ctx.move_to(center[0] - radius, center[1])
    ctx.arc(center[0], center[1], radius, -pi, pi)
    ctx.set_source_rgba(color[0], color[1], color[2], 0.9)
    ctx.fill()
    ctx.stroke()

def nPointSeed(image, n):
    "Seed according to the distribution in the image."
    # Compute a CDF function across the flattened image
    imageCDF = image.flatten().cumsum()
    imageCDF /= 1.0 * imageCDF.max()

    # Function to turn a random point in the CDF into a random index in the image
    indexInterpolator = interpolate.interp1d(imageCDF, arange(imageCDF.size))

    # Set to collect the UNIQUE indices
    indexContainer = set()
    while len(indexContainer) < n:
        # Generate at most the number of points remaining
        maxToGenerate = n - len(indexContainer)
        randomCDFValues = random.uniform(np.finfo(np.float32).eps, 1.0-np.finfo(np.float32).eps, maxToGenerate)

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

def optimize(generators, centroids, n, shape, radius):
    "Optimization using centroids of weighted Vornoi diagrams."
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
        if (np.abs(generators - centroids)).max() < radius:
            break

        generators[:] = np.round(centroids.copy())
    return centroids


def distance(l):
    return sum([L2[l[i-1], l[i]] for i in range(len(l))])

def twoOptSwap(route, i, k):
    delta = (L2[route[i-1], route[k-1]] + L2[route[i], route[k]]
            -L2[route[i-1], route[i]] -L2[route[k-1], route[k]])
    
    if delta > 0.1:
        return route
    else:
        new_route = route[:]
        new_route[i:k] = route[i:k][::-1]
        return new_route

def tsp():
    # calculate metric matrix
    nPoints = len(points)
    global L2
    L2 = np.zeros([nPoints, nPoints], dtype = np.float32)
    for i in range(nPoints):
        for j in range(i):
            L2[i, j] = L2[j, i] = np.sqrt(sum((points[i].p - points[j].p)**2))
    print "metric created!"

    # 2opt
    best_route = range(nPoints)
    best_distance = distance(best_route)
    route = range(nPoints)
    for count in range(int(np.sqrt(nPoints))):
        for iInterp in range(nPoints**2 / 2):
            i = iInterp // nPoints
            k = iInterp %  nPoints
            if k != i:
                i, k = sorted([i, k])
                route = twoOptSwap(route, i, k)

        route_distance = distance(route)
        if best_distance > route_distance+1:
             best_route, best_distance = route[:], route_distance
             print route_distance
        else:
            break

    return [points[i] for i in best_route]


if __name__=='__main__':
    # get the image to be stippled
    filename = "smiley.png"
    img = misc.imread(filename)
    global points
    points = []
    radius = 1.0
    K = 1

    # throw away alpha channel

    if (len(img.shape) == 3 and (K > 1)):
        # if color image cluster it to K colors 
        carray, frequency = colorCluster(img, K)
    else:  # black stipples only
        K = 1
        carray = np.array([[0.0] * 3])
        frequency = [mean(img)/255.0 * img.shape[0] * img.shape[1]]
        
    for i in range(K):
        # project the image along a color vector
        col = carray[i]
        gray_img = projectImage(img.astype(np.float32), col, 0.5).squeeze()

        # get the dimensions of image and number of points to replace it with
        imshape = gray_img.shape
        # nPoints = int(float(frequency[i]) / 50.0)
        nPoints = 20000

        # initialize generators and centroids arrays
        generators =  np.zeros([2, nPoints], dtype = np.int)
        centroids = np.zeros([2, nPoints], dtype = np.float32)

        # seed generators and optimize centroids
        generators = nPointSeed(gray_img, nPoints)
        centroids = optimize(generators, centroids, nPoints, imshape, radius)

        # gimme some numbers
        np.set_printoptions(precision=3)
        print "{0:.3f}".format(mean(gray_img)), col, nPoints, "stippled!"

        # append the result to the list of points:
        for cen in centroids.T:
            points.append(Point(cen, col))

    # randomise order of points to plot
    # random.shuffle(points)

    # sort the list in case that is faster
    sorted(points, key=lambda x: -sum(x.p**2))

    # tsp through these points.
    tsp_pts = []
    tsp_pts = tsp()

    # Make a pdf surface
    surf =  cairo.PDFSurface(open(filename.split(".")[0]+str(K)+'point.pdf',
                                  "w"), img.shape[1], img.shape[0])
    # Make a svg surface
    # surf =  cairo.SVGSurface(open("test.svg", "w"), img.shape[0], img.shape[1])
    
    # Get a context object and set line width
    ctx = cairo.Context(surf)
    ctx.set_line_width(0.5)

    xCoords = []
    yCoords = []
    ctx.move_to(tsp_pts[-1].p[1], tsp_pts[-1].p[0])
    # Make a data file
    with  open(filename.split(".")[0]+'-data.csv', "w") as datafile:
        datafile.write("d,x,y,r,g,b\n")
        for point in tsp_pts:
            # drawCircle(ctx, point.p, radius, point.c/255.0)
            ctx.set_source_rgb(point.c[0], point.c[1], point.c[2])
            ctx.line_to(point.p[1], point.p[0])

            datafile.write(str(radius) + "," + 
                           str(point.p[0]) + "," + 
                           str(point.p[1]) + "," +
                           str(int(point.c[0])) + "," +
                           str(int(point.c[1])) + "," +
                           str(int(point.c[2])) + "\n")
            xCoords.append(-point.p[0])
            yCoords.append(-point.p[1])
    ctx.close_path()
    ctx.fill()
    surf.finish()

    # plot the data
    pl.figure(1)
    pl.plot(yCoords, xCoords, 'g-')
    frame = pl.gca()
    frame.axes.get_xaxis().set_visible(False)
    frame.axes.get_yaxis().set_visible(False)
    pl.savefig('tsp-' + filename, bbox_inches='tight')
    pl.show()

    print len(points), " points polotted!"
