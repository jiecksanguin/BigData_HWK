
import math 
from pyspark import SparkContext, SparkConf
import sys
import time


'''
Let 𝑆 be a set of 𝑁 points from some metric space and, for each 𝑝∈𝑆 let 𝐵𝑆(𝑝,𝑟) denote the set of points of 𝑆 at distance 
at most 𝑟 from 𝑝. For given parameters 𝑀,𝐷>0, an (𝑀,𝐷) -outlier (w.r.t. 𝑆) is a point 𝑝∈𝑆 such that |𝐵𝑆(𝑝,𝐷)|≤𝑀. 
The problem that we want to study is the following: given 𝑆,𝑀, and 𝐷, mark each point 𝑝∈𝑆 as outlier, if it is an 
(𝑀,𝐷)-outlier, and non-outlier otherwise.
'''

def exactOutliers(listOfPoints, D, M, K):
    
    #check that all the values taken as argument are of the correct type
    if not isinstance(listOfPoints, list):
        raise TypeError("listOfPoints must be a list")
    if not isinstance(D, float):
        raise TypeError("D must be a float")
    if not isinstance(M, int):
        raise TypeError("M must be an integer")
    if not isinstance(K, int):
        raise TypeError("K must be a float")
    
    #complexity: O(𝑁(𝑁−1)/2)
    
    # Initialize an empty list to store outliers
    outliers = []

    # Initialize a counter list with all elements set to 1,
    # indicating each point has at least one neighbor (itself)
    counter = [1] * len(listOfPoints)
    # Loop through each pair of points to calculate distances and update the counter
    for i in range(len(listOfPoints)-1):
        # Get the coordinates of the first point
        p1 = listOfPoints[i]
        for j in range(i+1, len(listOfPoints)):
            # Check if the points are distinct
            if i != j:
                # Get the coordinates of the second point
                p2 = listOfPoints[j]
                # Calculate the Euclidean distance between the points
                distance = math.dist(p1, p2)
                # If the distance is less than or equal to D
                # increment the counter for both points
                if distance <= D:
                    counter[i] += 1
                    counter[j] += 1

    # Iterate over the indices and counts in the counter list
    for i, count in enumerate(counter):
        # Check if the count is less than or equal to M
        if count <= M:
            # If so, add the point and its count to the outliers list
            outliers.append((listOfPoints[i], count))

    #print the total number of outliers
    print("Printing the total number of outliers:", len(outliers))

    # Sort the outliers list so that it will have the outlier points in non-decreasing order of |𝐵𝑆(𝑝,𝐷)|
    sortedOutliers = sorted(outliers, key=lambda x: x[1])

    # Print only the first K outliers, one per line
    for point, _ in sortedOutliers[:K]:
        print("Point:", point)

    
def getCell(point, cellSideLength):
    i = math.floor(point[0] / cellSideLength)
    j = math.floor(point[1] / cellSideLength)
    return (i, j)

def gatherPairsPartitions(pairs):
    pairs_dict = {}
    for p in pairs:
        if p not in pairs_dict.keys():
            pairs_dict[p] = 1
        else:
            pairs_dict[p] += 1
    return [(key, pairs_dict[key]) for key in pairs_dict.keys()]


def MRApproxOutliers(points, D, M, K):
    # STEP A
    mapped_points1 = points.map(lambda x: getCell(x, D/(2 * math.sqrt(2)))) # Round 1
    mapped_points2 = mapped_points1.mapPartitions(gatherPairsPartitions)
    
    #2 possibilities:
    #mapped_points3 = mapped_points2.groupByKey()
    #mapped_points4 = mapped_points3.mapValues(lambda vals: sum(vals)).cache()    
                   
    mapped_points3 = mapped_points2.reduceByKey(lambda a, b: a + b)
    mapped_points4 = mapped_points3.cache()

    # Step B 


if __name__ == "__main__":
    # Check if the correct number of command-line arguments are provided
    if len(sys.argv) != 6:
        print("Usage: python test_exact_outliers.py <path_to_file> <D> <M> <K> <L>")
        sys.exit(1)

    # Extract command-line arguments
    path_to_file = sys.argv[1]
    D = float(sys.argv[2])
    M = int(sys.argv[3])
    K = int(sys.argv[4])
    L = int(sys.argv[5])

    print("Command-line arguments:")
    print("Path to file:", path_to_file)
    print("D:", D)
    print("M:", M)
    print("K:", K)
    print("L:", L)

    # Initialize SparkContext
    sc = SparkContext(appName="ExactOutliers")

    # Read input points into an RDD of strings (rawData)
    rawData = sc.textFile(path_to_file)

    # Transform rawData into an RDD of points (inputPoints), represented as pairs of floats
    inputPoints = rawData.map(lambda line: [float(x) for x in line.strip().split(',')])

    # Repartition inputPoints into L partitions
    inputPoints = inputPoints.repartition(L)

    # Print the total number of points
    total_points = inputPoints.count()
    print("Total number of points:", total_points)

    # Check if the number of points is at most 200000
    if total_points <= 200000:
        # Download the points into a list called listOfPoints
        listOfPoints = inputPoints.collect()

        # Execute ExactOutliers with parameters listOfPoints, D, M, and K
        start_time = time.time()
        exactOutliers(listOfPoints, D, M, K)
        end_time = time.time()
        print("ExactOutliers running time:", end_time - start_time)

    # Stop SparkContext
    sc.stop()
