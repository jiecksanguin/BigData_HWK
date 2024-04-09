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

def calculateN3N7(cellSizes):
    N3_N7_results = []
    for cell, size in cellSizes.items():
        i, j = cell
        N3 = 0
        N7 = 0
        # Iterate over a 7x7 grid around the cell
        for di in range(-3, 4):
            for dj in range(-3, 4):
                ni = i + di
                nj = j + dj
                if (ni, nj) in cellSizes:
                    cell_size = cellSizes[(ni, nj)]
                    if abs(di) <= 1 and abs(dj) <= 1:
                            N3 += cell_size
                    N7 += cell_size
        N3_N7_results.append((cell, N3, N7, cellSizes[cell]))
    return N3_N7_results


def MRApproxOutliers(points, D, M, K):
    
    cellSideLength = D/(2 * math.sqrt(2))
    print(cellSideLength)
    # STEP A
    mapped_points = (points.map(lambda x: getCell(x,cellSideLength)) 
        .mapPartitions(gatherPairsPartitions) 
        .reduceByKey(lambda a, b: a + b) 
        .cache())
    
    #2 possibilities:
    #.groupByKey()
    #.mapValues(lambda vals: sum(vals)).cache()   
    

    cellSizes = mapped_points.collectAsMap()

    N3_N7_results = calculateN3N7(cellSizes)
    
    # Calculate the number of sure outliers, uncertain points
    sure_outliers_count = sum(size for cell, N3 , N7, size in N3_N7_results if N7 <= M)
    uncertain_points_count = sum(size for cell, N3, N7, size in N3_N7_results if N3 <= M and N7 > M)
    smallest_cells = sorted(N3_N7_results, key=lambda x: x[1])[:K]
    
    # Print the K smallest non-empty cells
    print("Number of sure (D, M)-outliers:", sure_outliers_count)
    print("Number of uncertain points:", uncertain_points_count)
    print("K smallest non-empty cells:")
    for cell, N3, N7, size in smallest_cells:
        print("Cell:", cell, "Size:", cellSizes[cell])


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
    sc = SparkContext(appName="Outliers")

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

    start_time_approx = time.time()
    MRApproxOutliers(inputPoints, D, M, K)
    end_time_approx = time.time()
    print("MRApproxOutliers running time:", end_time_approx - start_time_approx, "seconds")

    
    # Stop SparkContext
    sc.stop()
