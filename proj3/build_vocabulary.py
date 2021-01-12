from PIL import Image
import numpy as np
from cyvlfeat.sift.dsift import dsift
from cyvlfeat.kmeans import kmeans
from time import time


# This function will sample SIFT descriptors from the training images,
# cluster them with kmeans, and then return the cluster centers.

def build_vocabulary(image_paths, vocab_size):
    bag_of_features = []

    print("Extract SIFT features")

    # The Python Debugger
    # pdb.set_trace()

    for path in image_paths:
        img = np.asarray(Image.open(path), dtype='float32')
        frames, descriptors = dsift(img, step=[5, 5], fast=True)
        bag_of_features.append(descriptors)
    bag_of_features = np.concatenate(bag_of_features, axis=0).astype('float32')
    # pdb.set_trace()

    print("Compute vocab")
    start_time = time()
    vocab = kmeans(bag_of_features, vocab_size, initialization="PLUSPLUS")
    end_time = time()
    print("It takes ", (start_time - end_time), " to compute vocab.")

    return vocab
