import sys
import random
from pyspark import SparkConf, SparkContext
from pyspark.streaming import StreamingContext
from pyspark import StorageLevel
import threading
import math
import time

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
def sticky_sampling(items, epsilon, sticky_frequency_map):
    sample_threshold = int(1 / epsilon)
    for item in items:
        if item in sticky_frequency_map:
            sticky_frequency_map[item] += 1
        elif len(sticky_frequency_map) < sample_threshold:
            sticky_frequency_map[item] = 1
        else:
            for key in list(sticky_frequency_map.keys()):
                sticky_frequency_map[key] -= 1
                if sticky_frequency_map[key] == 0:
                    del sticky_frequency_map[key]

# Operations to perform after receiving an RDD 'batch' at time 'time'
def process_batch(time, batch, streamLength, n, phi, epsilon, histogram, reservoir, sticky_frequency_map, stopping_condition):
    batch_size = batch.count()
    # If we already have enough points (> n), skip this batch.
    if streamLength[0] >= n:
        return
    streamLength[0] += batch_size
    
    # Extract items from the batch
    batch_items = batch.map(lambda s: int(s)).collect()

    # Update the histogram for true frequent items
    for item in batch_items:
        if item in histogram:
            histogram[item] += 1
        else:
            histogram[item] = 1

    # Apply Reservoir Sampling
    reservoir_sampling(batch_items, m, reservoir)

    # Apply Sticky Sampling
    sticky_sampling(batch_items, epsilon, sticky_frequency_map)

    if streamLength[0] >= n:
        stopping_condition.set()

def main():
    # Parse command-line arguments
    if len(sys.argv) != 6:
        print("Usage: GxxxHW3.py <n> <phi> <epsilon> <delta> <portExp>")
        sys.exit(-1)

    n = int(sys.argv[1])
    phi = float(sys.argv[2])
    epsilon = float(sys.argv[3])
    delta = float(sys.argv[4])
    portExp = int(sys.argv[5])

    # Constants for reservoir sampling
    global m
    m = math.ceil(1 / phi)

    # Data structures to maintain the state
    streamLength = [0]  # Stream length (an array to be passed by reference)
    histogram = {}  # Hash Table for the distinct elements
    reservoir = []  # Reservoir for reservoir sampling
    sticky_frequency_map = {}  # Frequency map for sticky sampling

    # Semaphore for clean shutdown
    stopping_condition = threading.Event()

    # Configure Spark
    conf = SparkConf().setMaster("local[*]").setAppName("GxxxHW3")
    sc = SparkContext(conf=conf)
    ssc = StreamingContext(sc, 0.01)  # Batch duration of 0.01 seconds
    ssc.sparkContext.setLogLevel("ERROR")

    # Create DStream from the specified socket
    stream = ssc.socketTextStream("algo.dei.unipd.it", portExp, StorageLevel.MEMORY_AND_DISK)
    # Process each batch
    stream.foreachRDD(lambda time, batch: process_batch(time, batch, streamLength, n, phi, epsilon, histogram, reservoir, sticky_frequency_map, stopping_condition))

    # Start the streaming context in a separate thread
    def start_streaming():
        try:
            ssc.start()
            ssc.awaitTermination()
        except Exception as e:
            print(f"Error in streaming context: {e}")

    streaming_thread = threading.Thread(target=start_streaming)
    streaming_thread.start()

    # Wait for the stopping condition to be set
    stopping_condition.wait()
    
    # Try to stop the streaming context gracefully
    def stop_streaming():
        try:
            ssc.stop(False, True)
        except Exception as e:
            print(f"Error stopping streaming context: {e}")

    stop_streaming_thread = threading.Thread(target=stop_streaming)
    stop_streaming_thread.start()
    stop_streaming_thread.join()

    # Ensure the streaming thread completes
    streaming_thread.join()

    # Compute true frequent items
    true_frequent_items = [item for item, count in histogram.items() if count >= phi * streamLength[0]]
    true_frequent_items.sort()

    # Print the results
    print(f"Input parameters: n={n}, phi={phi}, epsilon={epsilon}, delta={delta}, portExp={portExp}")

    # Exact Algorithm Results
    print(f"Size of the data structure used to compute the true frequent items: {len(histogram)}")
    print(f"Number of true frequent items: {len(true_frequent_items)}")
    print("True frequent items:")
    for item in true_frequent_items:
        print(item)
        
    # Reservoir Sampling Results
    reservoir.sort()
    estimated_frequent_items = sorted(set(reservoir))
    print(f"Size m of the Reservoir sample: {m}")
    print(f"Number of estimated frequent items: {len(estimated_frequent_items)}")
    print("Estimated frequent items (Reservoir Sampling):")
    for item in estimated_frequent_items:
        if item in true_frequent_items:
            print(f"{item} +")
        else:
            print(f"{item} -")

    # Sticky Sampling Results
    print(f"Size of the Hash Table: {len(sticky_frequency_map)}")
    approx_frequent_items = sorted(sticky_frequency_map.keys())
    print(f"Number of estimated frequent items (Sticky Sampling): {len(approx_frequent_items)}")
    print("Epsilon-Approximate Frequent Items (Sticky Sampling):")
    for item in approx_frequent_items:
        if item in true_frequent_items:
            print(f"{item} +")
        else:
            print(f"{item} -")

if __name__ == "__main__":
    main()