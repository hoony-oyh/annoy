from annoy import AnnoyIndex
import fastannoy
import random

index_file = "../../index/dot_mnist.ann"
f = 784

u = fastannoy.AnnoyIndex(f, "dot")
u.load(index_file)

random_query = [random.randint(0, 10000) for _ in range(10)]
k = 10

for q in random_query:
    result = u.get_nns_by_item(q, k) # will find the 10 nearest neighbors
    print(q, result)
