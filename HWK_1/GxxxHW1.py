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

#TEST
#supposed output should be (3, 3) (5, 5) (0, 0), one per line
points = [(0, 0), (0, 1), (1, 0), (3, 3), (4, 4), (5, 5)]
D = 2.5
M = 2
K = 3
exactOutliers(points, D, M, K) 

