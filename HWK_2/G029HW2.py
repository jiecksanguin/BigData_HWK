import math 
from pyspark import SparkContext, SparkConf
import sys
import time

    
def get_cell(point, cell_side_length):
    # Calculate the cell coordinates (i, j) for a given point
    # by dividing its coordinates by the cell side length
    i = math.floor(point[0] / cell_side_length)
    j = math.floor(point[1] / cell_side_length)
    return (i, j)


def gather_pairs_partitions(pairs):
    # Create a dictionary to count occurrences of each pair
    pairs_dict = {}
    for p in pairs:
        if p not in pairs_dict.keys():
            pairs_dict[p] = 1
        else:
            pairs_dict[p] += 1
    # Convert the dictionary to a list of key-value pairs
    return [(key, pairs_dict[key]) for key in pairs_dict.keys()]

def calculate_N3_N7(cell_sizes):
    # Calculate the N3 and N7 metrics for each cell based on neighboring cells
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
                # Check if the neighboring cell exists in cell_sizes
                if (ni, nj) in cell_sizes:
                    cell_size = cell_sizes[(ni, nj)]
                    # Update N3 if the neighboring cell is within 1 unit distance
                    if abs(di) <= 1 and abs(dj) <= 1:
                            N3 += cell_size
                    # Always update N7 for any neighboring cell
                    N7 += cell_size
        # Append the computed N3, N7, and cell size to the results
        N3_N7_results.append((cell, N3, N7, cell_sizes[cell]))
    return N3_N7_results

def SequentialFFT(P, K):
    # Initialize an empty list to store the centers
    C = []
    PP = P[:]
    # Choose an arbitrary point from P as the first center
    farthest_point = P[0]
    C.append(farthest_point)
    #PP.remove(farthest_point)

    distance_dict = {}
    for point in PP:
        distance_dict[tuple(point)] = math.dist(point, farthest_point)
    del distance_dict[tuple(farthest_point)]
    while len(C) < K:
        next_center = max(distance_dict.keys(), key=lambda x: distance_dict[x])

        for point in distance_dict.keys():
            current_distance = math.dist(point, next_center)
            if current_distance < distance_dict[tuple(point)]:
                distance_dict[tuple(point)] = current_distance
        C.append(next_center)
        del distance_dict[tuple(next_center)]
        #PP.remove(next_center)
    return C 

def MRFFT(inputPoints, K):
    # Round 1: MR-FarthestFirstTraversal
    start_time_round1 = time.time()
    coreset= inputPoints.mapPartitions(lambda partition: [SequentialFFT(list(partition), K)]).flatMap(lambda x: x).collect()
    end_time_round1 = time.time()
    milliseconds_round1 = (end_time_round1 - start_time_round1) * 1000
    print("Running time of Round 1 =", milliseconds_round1, "ms")

    # Round 2: SequentialFFT on coreset
    start_time_round2 = time.time()
    centers = SequentialFFT(coreset, K)
    end_time_round2 = time.time()
    milliseconds_round2 = (end_time_round2 - start_time_round2) * 1000
    print("Running time of Round 2 =", milliseconds_round2, "ms")

    # Broadcast centers for Round 3
    centers_broadcast = sc.broadcast(centers)
    
    # Round 3: Compute radius R
    start_time_round3 = time.time()
    R = inputPoints.map(lambda x: min([math.dist(x, c) for c in centers_broadcast.value])).reduce(max)
    end_time_round3 = time.time()
    milliseconds_round3 = (end_time_round3 - start_time_round3) * 1000
    print("Running time of Round 3 =", milliseconds_round3, "ms")

    return R


def MRApproxOutliers(points, D, M):

    cell_side_length = D/(2 * math.sqrt(2))

    # STEP A: Map each point to a cell, gather and count pairs within each partition
    mapped_points = (points.map(lambda x: get_cell(x,cell_side_length)) 
        .mapPartitions(gather_pairs_partitions) 
        .reduceByKey(lambda a, b: a + b) 
        .cache())
    
    #STEP B: Collect cell sizes as a map and calculate N3/N7 metrics
    cellSizes = mapped_points.collectAsMap()
    N3_N7_results = calculate_N3_N7(cellSizes)

    # Calculate the number of sure outliers, uncertain points
    sure_outliers_count = sum(size for cell, N3 , N7, size in N3_N7_results if N7 <= M)
    uncertain_points_count = sum(size for cell, N3, N7, size in N3_N7_results if N3 <= M and N7 > M)
    
    print("Number of sure outliers=", sure_outliers_count)
    print("Number of uncertain points=", uncertain_points_count)


if __name__ == "__main__":
    # Check if the correct number of command-line arguments are provided
    if len(sys.argv) != 5:
        print("Usage: python test_exact_outliers.py <path_to_file> <M> <K> <L>")
        sys.exit(1)

    # Extract command-line arguments
    path_to_file = sys.argv[1]
    M = int(sys.argv[2])
    K = int(sys.argv[3])
    L = int(sys.argv[4])

    print(f"{path_to_file} M={M} K={K} L={L}")

    # Initialize SparkContext
    sc = SparkContext(appName="Outliers + Clustering")

    # Read input points into an RDD of strings (rawData)
    rawData = sc.textFile(path_to_file)

    # Transform rawData into an RDD of points (inputPoints), represented as pairs of floats
    inputPoints = rawData.map(lambda line: [float(x) for x in line.strip().split(',')])

    # Repartition inputPoints into L partitions
    inputPoints = inputPoints.repartition(L)

    # Print the total number of points
    total_points = inputPoints.count()
    print("Number of points =", total_points)

    # Execute MRFFT to get the radius
    D = MRFFT(inputPoints, K)
    
    print("Radius =", D)
    
    start_time_approx = time.time()
    MRApproxOutliers(inputPoints, D, M)
    end_time_approx = time.time()
    milliseconds_approx = (end_time_approx - start_time_approx) * 1000
    print("Running time of MRApproxOutliers =", milliseconds_approx, "ms")

    
    # Stop SparkContext
    sc.stop()
