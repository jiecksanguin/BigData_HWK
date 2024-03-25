
import math

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
    
    #for each point, check the distance between that point and all the other points. 
    #If the distance is <= D add a plus 1 to a counter. 
    #If the counter doesn't get to a value which is greater than M, then consider the point as an outlier.
    outliers = []
    for i in range(len(listOfPoints)):
        counter = 0
        p1 = listOfPoints[i]

        for j in range(len(listOfPoints)):
            if i != j:
                p2 = listOfPoints[j]
                distance = math.dist(p1, p2)
                
                if distance <= D:
                    counter += 1
                    
        if counter <= M:
            outliers.append((p1, counter))
    
    #sort the outliers list so that it will have the outlier points in non-decreasing order of |𝐵𝑆(𝑝,𝐷)|
    sortedOutliers = sorted(outliers, key=lambda x: x[1], reverse=False)

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
