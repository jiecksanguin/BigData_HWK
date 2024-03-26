import math 
from pyspark import SparkContext
"""
Let 𝑆 be a set of 𝑁 points from some metric space and, for each 𝑝∈𝑆 let 𝐵𝑆(𝑝,𝑟) denote the set of points of 𝑆 at distance 
at most 𝑟 from 𝑝. For given parameters 𝑀,𝐷>0, an (𝑀,𝐷) -outlier (w.r.t. 𝑆) is a point 𝑝∈𝑆 such that |𝐵𝑆(𝑝,𝐷)|≤𝑀. 
The problem that we want to study is the following: given 𝑆,𝑀, and 𝐷, mark each point 𝑝∈𝑆 as outlier, if it is an 
(𝑀,𝐷)-outlier, and non-outlier otherwise.

"""
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
    
    #compute all the pairwise distances
    #complexity: O(𝑁(𝑁−1)/2)
    distances = {}
    for i in range(len(listOfPoints)):
        for j in range(i+1,len(listOfPoints)):
            p1 = listOfPoints[i]
            p2 = listOfPoints[j]
            distance = math.dist(p1, p2)
            distances[(i,j)] = distance

    outliers = []
    for i, point in enumerate(listOfPoints):
        counter = 0
        p1 = listOfPoints[i]
        for j, other_point in enumerate(listOfPoints):
            if i != j:  # Exclude distance to itself
                if (i, j) in distances:
                    distance = distances[(i, j)]
                elif (j, i) in distances:
                    distance = distances[(j, i)]
                else:
                    raise ValueError(f"No distance found between {point} and {other_point}")

                if distance <= D:
                    counter += 1

        if counter <= M:
            outliers.append((p1, counter))

    #print(outliers)
    #sort the outliers list so that it will have the outlier points in non-decreasing order of |𝐵𝑆(𝑝,𝐷)|
    sortedOutliers = sorted(outliers, key=lambda x: x[1], reverse=False)

    #print(sortedOutliers)
    # Print only the first K outliers, one per line
    for tuple in sortedOutliers[:K]:
        print(tuple[0])


def cell_identifier(point, cell_side_length):
    i = int(point[0] // cell_side_length)
    j = int(point[1] // cell_side_length)
    return (i, j)

def count_points_in_cell(iterator):
    counts = {}
    for point in iterator:
        cell = cell_identifier(point, cell_side_length)
        counts[cell] = counts.get(cell, 0) + 1
    return counts.items()

def count_neighbors(cell, cell_counts):
    i, j = cell
    neighbors_count = 0
    for x in range(i-1, i+2):
        for y in range(j-1, j+2):
            neighbors_count += cell_counts.get((x, y), 0)
    return neighbors_count

def MRApproxOutliers(points_rdd, D, M, K):
    # Step A: Count points in each cell
    cell_side_length = D / (2 * 2**0.5)
    cell_counts = points_rdd.mapPartitions(count_points_in_cell) \
                            .reduceByKey(lambda x, y: x + y)
    
    # Step B: Attach N3 and N7 to each non-empty cell
    cell_N3_N7 = cell_counts.map(lambda cell_count: (cell_count[0], cell_count[1], 
                                                      count_neighbors(cell_count[0], cell_counts)))
    
    # Collect small cell counts in local memory
    cell_N3_N7_local = cell_N3_N7.collect()

    # Compute sure outliers, uncertain points, and K smallest cells
    sure_outliers = cell_N3_N7.filter(lambda cell: cell[2] <= M).count()
    uncertain_points = cell_N3_N7.filter(lambda cell: cell[1] > M and cell[2] > M).count()
    smallest_cells = sorted(cell_N3_N7_local, key=lambda x: x[1])[:K]

    # Print results
    print("Sure (D, M)-outliers:", sure_outliers)
    print("Uncertain points:", uncertain_points)
    print("Smallest non-empty cells (identifier, size):", smallest_cells)

# Example usage
if __name__ == "__main__":
    sc = SparkContext("local", "MRApproxOutliers")
    points_rdd = sc.parallelize([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0), (9.0, 10.0)])
    D = 10.0
    M = 2
    K = 3
    MRApproxOutliers(points_rdd, D, M, K)
    sc.stop()


#TEST
#supposed output should be (3, 3) (5, 5) (0, 0), one per line
points = [(0, 0), (0, 1), (1, 0), (3, 3), (4, 4), (5, 5)]
D = 2.5
M = 2
K = 3
exactOutliers(points, D, M, K) 

