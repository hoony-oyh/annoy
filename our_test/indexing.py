import numpy as np
from annoy import AnnoyIndex
import fastannoy
import time

f = 256
tree_num = 3
metric = 'dot'
random_seed = 645
run_iter = 5

F = [128, 256, 512]
N = [10000, 30000, 100000] #, 300000, 1000000]

for n in N:
	for f in F:
		print("="*50)
		print("Number of data: {} / Dimension: {}".format(n, f))
		print("Ours")
		t = fastannoy.AnnoyIndex(f, metric)
		t.set_seed(random_seed)

		data = []
		for i in range(n):
			data.append(np.random.randn(f).tolist())

		for i in range(n):
			t.add_item(i, data[i])
		
		total_time = 0
		for j in range(run_iter):
			t.unbuild()
			start_time = time.time()
			t.build(tree_num)
			end_time = time.time()
			total_time += (end_time - start_time)
	
		result = t.get_nns_by_item(0, 10)
		print(result)
		print("Build time:", total_time/run_iter)
		t.save("index/{}_{}_{}.ann".format(metric, f, n))

		print("\n\nBaseline")
		t2 = AnnoyIndex(f, metric)
		t2.set_seed(random_seed)
		for i in range(n):
			t2.add_item(i, data[i])

		total_time = 0
		for j in range(run_iter):
			t2.unbuild()
			start_time = time.time()
			t2.build(tree_num);
			end_time = time.time()
			total_time += (end_time - start_time)

		result2 = t2.get_nns_by_item(0, 10)
		print(result2)
		print("Build time:", total_time/run_iter)
		print("\n")
