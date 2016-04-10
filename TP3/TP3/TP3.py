import matplotlib.cm as cm
import matplotlib.pyplot as plt
import gzip
import pickle
import numpy
import NeuralNetwork

def main():
    with gzip.open('mnist.pkl.gz', 'rb') as file:
        train_set, valid_set, test_set = pickle.load(file, encoding='latin1')

    train_x, train_y = train_set

    neuralNetwork = NeuralNetwork.NeuralNetwork([784, 30, 30, 30, 10])
    neuralNetwork.stochatstic_gradient_descent(train_set, valid_set, test_set, 500, 1.5)

    #plt.imshow(train_x[12].reshape((28,28)), cmap = cm.Greys_r)
    #plt.show()

if __name__ == "__main__":
    main()
