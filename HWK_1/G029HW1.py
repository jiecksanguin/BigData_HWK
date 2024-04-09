import math 
from pyspark import SparkContext, SparkConf
import sys
import time


def exactOutliers(list_of_points, D, M, K):
    
    #complexity: O(𝑁(𝑁−1)/2)
    
    # Initialize an empty list to store outliers
    outliers = []

    # Initialize a counter list with all elements set to 1,
    # indicating each point has at least one neighbor (itself)
    counter = [1] * len(list_of_points)
    # Loop through each pair of points to calculate distances and update the counter
    for i in range(len(list_of_points)-1):
        # Get the coordinates of the first point
        p1 = list_of_points[i]
        for j in range(i+1, len(list_of_points)):
            # Check if the points are distinct
            if i != j:
                # Get the coordinates of the second point
                p2 = list_of_points[j]
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
            outliers.append((list_of_points[i], count))

    #print the total number of outliers
    print("Number of Outliers =", len(outliers))

    # Sort the outliers list so that it will have the outlier points in non-decreasing order of |𝐵𝑆(𝑝,𝐷)|
    sorted_outliers = sorted(outliers, key=lambda x: x[1])

    # Print only the first K outliers, one per line
    for point, _ in sorted_outliers[:K]:
        print("Point:", f"({point[0]}, {point[1]})")

    
def get_cell(point, cell_side_length):
    i = math.floor(point[0] / cell_side_length)
    j = math.floor(point[1] / cell_side_length)
    return (i, j)

def gather_pairs_partitions(pairs):
    pairs_dict = {}
    for p in pairs:
        if p not in pairs_dict.keys():
            pairs_dict[p] = 1
        else:
            pairs_dict[p] += 1
    return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

def calculate_N3_N7(cell_sizes):
    N3_N7_results = []
    for cell, size in cell_sizes.items():
        i, j = cell
        N3 = 0
        N7 = 0
        # Iterate over a 7x7 grid around the cell
        for di in range(-3, 4):
            for dj in range(-3, 4):
                ni = i + di
                nj = j + dj
                if (ni, nj) in cell_sizes:
                    cell_size = cell_sizes[(ni, nj)]
                    if abs(di) <= 1 and abs(dj) <= 1:
                            N3 += cell_size
                    N7 += cell_size
        N3_N7_results.append((cell, N3, N7, cell_sizes[cell]))
    return N3_N7_results


def MRApproxOutliers(points, D, M, K):
    
    cell_side_length = D/(2 * math.sqrt(2))

    # STEP A
    mapped_points = (points.map(lambda x: get_cell(x,cell_side_length)) 
        .mapPartitions(gather_pairs_partitions) 
        .reduceByKey(lambda a, b: a + b) 
        .cache())
    
    #STEP B
    cellSizes = mapped_points.collectAsMap()
    N3_N7_results = calculate_N3_N7(cellSizes)

    # Calculate the number of sure outliers, uncertain points
    sure_outliers_count = sum(size for cell, N3 , N7, size in N3_N7_results if N7 <= M)
    uncertain_points_count = sum(size for cell, N3, N7, size in N3_N7_results if N3 <= M and N7 > M)
    smallest_cells = sorted(N3_N7_results, key=lambda x: x[1])[:K]
    
    # Print the K smallest non-empty cells
    print("Number of sure outliers=", sure_outliers_count)
    print("Number of uncertain points=", uncertain_points_count)
    for cell, N3, N7, size in smallest_cells:
        print("Cell:", cell, "Size =", cellSizes[cell])


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

    print(f"{path_to_file} D={D} M={M} K={K} L={L}")

    # Initialize SparkContext
    sc = SparkContext(appName="Outliers")

    # Read input points into an RDD of strings (rawData)
    raw_data = sc.textFile(path_to_file)

    # Transform rawData into an RDD of points (inputPoints), represented as pairs of floats
    input_points = raw_data.map(lambda line: [float(x) for x in line.strip().split(',')])

    # Repartition inputPoints into L partitions
    input_points = input_points.repartition(L)

    # Print the total number of points
    total_points = input_points.count()
    print("Number of points =", total_points)

    # Check if the number of points is at most 200000
    if total_points <= 200000:
        # Download the points into a list called listOfPoints
        list_of_points = input_points.collect()

        # Execute ExactOutliers with parameters listOfPoints, D, M, and K
        start_time_exact = time.time()
        exactOutliers(list_of_points, D, M, K)
        end_time_exact = time.time()
        milliseconds_exact = (end_time_exact - start_time_exact) * 1000
        print("Running time of ExactOutliers =", milliseconds_exact, "ms")

    start_time_approx = time.time()
    MRApproxOutliers(input_points, D, M, K)
    end_time_approx = time.time()
    milliseconds_approx = (end_time_approx - start_time_approx) * 1000
    print("Running time of MRApproxOutliers =", milliseconds_approx, "ms")

    
    # Stop SparkContext
    sc.stop()
