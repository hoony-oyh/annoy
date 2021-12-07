import matplotlib.pyplot as plt
import idx2numpy

if __name__ == "__main__":
    answer = [(8007, [8111, 2649, 7987, 8023, 7838, 7913, 7846, 8029, 4804]),
              (9477, [7840, 7815, 7894, 7772, 7781, 7935, 7785, 7994, 7739]),
              (1661, [68, 1480, 1114, 7312, 7329, 6116, 8471, 8080, 8518]),
              (5123, [5104, 7751, 3971, 6734, 9537, 5792, 8054, 7976, 8886]),
              (1888, [8103, 8096, 3402, 4619, 834, 1916, 2620, 8104, 8063, 7768]),
              (9147, [7844, 6069, 9164, 9169, 9393, 4539, 8786, 413, 9067]),
              (2233, [9068, 9092, 1613, 9088, 6004, 4246, 3521, 8778, 9110]),
              (928, [5377, 1853, 4158, 6228, 99, 4553, 1787, 9404, 8467])]

    imagefile = '../data/t10k-images-idx3-ubyte'
    imagearray = idx2numpy.convert_from_file(imagefile)

    ri = 0
    f, axarr = plt.subplots(8, 6, figsize=(50, 50))
    for a, prediction in answer:
        axarr[ri][0].imshow(imagearray[a], cmap=plt.cm.binary)
        for i, p in enumerate(prediction[:5]):
            axarr[ri][i + 1].imshow(imagearray[p], cmap=plt.cm.binary)
        ri += 1
    plt.axis('off')
    plt.show()

    print("hello")
