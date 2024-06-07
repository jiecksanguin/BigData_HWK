import sys
import random
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import math

# Reservoir Sampling function
def reservoir_sampling(items, k, reservoir):
    count = len(reservoir)
    for item in items:
        count += 1
        if len(reservoir) < k:
            reservoir.append(item)
        else:
            s = random.randint(0, count - 1)
            if s < k:
                reservoir[s] = item

# Sticky Sampling function
def sticky_sampling(items, n, r, sticky_frequency_map):
    sample_probability = r / n
    for item in items:
        if item in sticky_frequency_map:
            sticky_frequency_map[item] += 1
        else:
            if random.random() < sample_probability:
                sticky_frequency_map[item] = 1

# Operations to perform after receiving an RDD 'batch' at time 'time'
def process_batch(time, batch, streamLength, n, m, r, histogram, reservoir, sticky_frequency_map, stopping_condition): 
    # Get the size of the current batch
    batch_size = batch.count()
    # Calculate the remaining number of items to process to reach n
    remaining_items = n - streamLength[0]
    
    # If the required number of items has already been processed, skip this batch
    if remaining_items <= 0:
        return
    
    # If the current batch size exceeds the remaining number of items, take only the necessary items
    if batch_size > remaining_items:
        batch_items = batch.map(lambda s: int(s)).take(remaining_items)
    else:
        batch_items = batch.map(lambda s: int(s)).collect()

    # Update the stream length with the number of processed items
    processed_count = len(batch_items)
    streamLength[0] += processed_count

    # Update the histogram for true frequent items
    for item in batch_items:
        if item in histogram:
            histogram[item] += 1
        else:
            histogram[item] = 1

    # Apply Reservoir Sampling
    reservoir_sampling(batch_items, m, reservoir)

    # Apply Sticky Sampling
    sticky_sampling(batch_items, n, r, sticky_frequency_map)

    if streamLength[0] >= n:
        stopping_condition.set()

def main():
    # Parse command-line arguments
    if len(sys.argv) != 6:
        print("Usage: G029HW3.py <n> <phi> <epsilon> <delta> <portExp>")
        sys.exit(-1)

    n = int(sys.argv[1])
    phi = float(sys.argv[2])
    epsilon = float(sys.argv[3])
    delta = float(sys.argv[4])
    portExp = int(sys.argv[5])

    # Constants for reservoir sampling
    m = math.ceil(1 / phi)

    # Calculate the sampling rate r
    r = math.log(1 / (delta * phi)) / epsilon

    # Data structures to maintain the state
    streamLength = [0]  # Stream length (an array to be passed by reference)
    histogram = {}  # Hash Table for the distinct elements
    reservoir = []  # Reservoir for reservoir sampling
    sticky_frequency_map = {}  # Frequency map for sticky sampling

    # Semaphore for clean shutdown
    stopping_condition = threading.Event()

    # Configure Spark
    conf = SparkConf().setMaster("local[*]").setAppName("G029HW3")
    conf = conf.set("spark.executor.memory", "4g").set("spark.driver.memory", "4g")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.01)  # Batch duration of 0.01 seconds
    ssc.sparkContext.setLogLevel("ERROR")

    # Create DStream from the specified socket
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    # Process each batch
    stream.foreachRDD(lambda time, batch: process_batch(time, batch, streamLength, n, m, r, histogram, reservoir, sticky_frequency_map, stopping_condition))

    # Start the streaming context
    ssc.start()

    # Wait for the stopping condition to be set
    stopping_condition.wait()
    
    # Stop the streaming context gracefully
    ssc.stop(False, True)

    # Compute true frequent items
    true_frequent_items = [item for item, count in histogram.items() if count >= phi * streamLength[0]]
    true_frequent_items.sort()

    # Print the results
    print(f"INPUT PROPERTIES")
    print(f"n = {n} phi = {phi} epsilon = {epsilon} delta = {delta} port = {portExp}")

    # Exact Algorithm Results
    print("EXACT ALGORITHM")
    print(f"Number of items in the data structure = {len(histogram)}")
    print(f"Number of true frequent items = {len(true_frequent_items)}")
    print("True frequent items:")
    for item in true_frequent_items:
        print(item)
        
    # Reservoir Sampling Results
    reservoir.sort()
    estimated_frequent_items = sorted(set(reservoir))
    print("RESERVOIR SAMPLING")
    print(f"Size m of the sample = {m}")
    print(f"Number of estimated frequent items = {len(estimated_frequent_items)}")
    print("Estimated frequent items:")
    for item in estimated_frequent_items:
        if item in true_frequent_items:
            print(f"{item} +")
        else:
            print(f"{item} -")

    # Sticky Sampling Results
    print("STICKY SAMPLING")
    print(f"Number of items in the Hash Table = {len(sticky_frequency_map)}")
    approx_frequent_items = [item for item, count in sticky_frequency_map.items() if count >= (phi - epsilon) * streamLength[0]]
    approx_frequent_items.sort()
    print(f"Number of estimated frequent items = {len(approx_frequent_items)}")
    print("Estimated frequent items:")
    for item in approx_frequent_items:
        if item in true_frequent_items:
            print(f"{item} +")
        else:
            print(f"{item} -")

if __name__ == "__main__":
    main()