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

def MRApproxOutliers(inputPoints, D, M, K):
    # Step A: Transform input RDD into RDD of cells with their counts
    def map_to_cells(point):
        x, y = point
        cell_i = int(x / (D / (2**0.5)))
        cell_j = int(y / (D / (2**0.5)))
        return ((cell_i, cell_j), 1)
    
    def add_counts(count1, count2):
        return count1 + count2
    
    cells_with_counts = inputPoints.map(map_to_cells).reduceByKey(add_counts)
    
    # Step B: Transform RDD of cells to include N3 and N7 counts
    def map_to_cell_info(cell_with_count):
        (cell_i, cell_j), count = cell_with_count
        N3 = cells_with_counts.filter(lambda x: abs(x[0][0] - cell_i) <= 1 and abs(x[0][1] - cell_j) <= 1).map(lambda x: x[1]).sum()
        N7 = cells_with_counts.filter(lambda x: abs(x[0][0] - cell_i) <= 3 and abs(x[0][1] - cell_j) <= 3).map(lambda x: x[1]).sum()
        return ((cell_i, cell_j), (count, N3, N7))
    
    cells_with_info = cells_with_counts.map(map_to_cell_info)
    
    # Count sure outliers, uncertain points, and find top K non-empty cells
    cells_info_collected = cells_with_info.collect()
    sure_outliers_count = len([cell_info for cell_info in cells_info_collected if cell_info[1][1] > M])
    uncertain_points_count = cells_with_info.filter(lambda x: x[1][1] <= M and x[1][2] > M).count()
    top_cells = cells_with_info.sortBy(lambda x: x[1][0], ascending=False).take(K)
    
    return sure_outliers_count, uncertain_points_count, top_cells

# Example usage:
if __name__ == "__main__":
    sc = SparkContext("local", "MRApproxOutliers")
    inputPoints = sc.parallelize([(1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)])
    D = 1.0
    M = 2
    K = 3
    sure_outliers_count, uncertain_points_count, top_cells = MRApproxOutliers(inputPoints, D, M, K)
    
    print("Sure outliers count:", sure_outliers_count)
    print("Uncertain points count:", uncertain_points_count)
    print("Top cells:", top_cells)

#TEST
#supposed output should be (3, 3) (5, 5) (0, 0), one per line
points = [(0, 0), (0, 1), (1, 0), (3, 3), (4, 4), (5, 5)]
D = 2.5
M = 2
K = 3
exactOutliers(points, D, M, K) 

