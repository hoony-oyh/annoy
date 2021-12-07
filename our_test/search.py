from annoy import AnnoyIndex
import fastannoy
import time
import random

run_iter = 50
num_element = 300
k = 50
random_list = [random.randint(0, 300000) for _ in range(num_element)]
metric = "dot"

print("="*50)
print("Baseline implementation")
for f in [256]:
        for N in [10000, 30000, 100000]:
                print("Number of data: {} / Dimension: {}".format(N, f))
                u = AnnoyIndex(f, metric)
                u.load("index/{}_{}_{}.ann".format(metric,f,N)) # super fast, will just mmap the file

                total_time = 0
                for _ in range(run_iter):
                        for i in range(num_element):
                                start_time = time.time()
                                result = u.get_nns_by_item(random_list[i] % N, k) # will find the 10 nearest neighbors
                                end_time = time.time()
                                total_time += (end_time - start_time)
                print(result)
                print("Average Query Time:", total_time/num_element/run_iter)
                print("\n")

print("="*50)
print("\nOur implementation")
for f in [256]:
        for N in [10000, 30000, 100000]:
                print("Number of data: {} / Dimension: {}".format(N, f))
                u = fastannoy.AnnoyIndex(f, metric)
                u.load("index/{}_{}_{}.ann".format(metric,f,N)) # super fast, will just mmap the file

                total_time = 0
                for _ in range(run_iter):
                        for i in range(num_element):
                                start_time = time.time()
                                result = u.get_nns_by_item(random_list[i] % N, k) # will find the 10 nearest neighbors
                                end_time = time.time()
                                total_time += (end_time - start_time)
                print(result)
                print("Average Query Time:", total_time/num_element/run_iter)
                print("\n")
