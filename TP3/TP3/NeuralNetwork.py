import numpy
import csv

class NeuralNetwork(object):
    def __init__(self, layer_sizes):
        self.number_of_layers = len(layer_sizes)
        self.layer_sizes = layer_sizes
        self.thetas = [numpy.random.rand(y, x + 1) / numpy.sqrt(y) for x, y in zip(layer_sizes[:-1], layer_sizes[1:])]
        self.a = 0.1

    def pw_linear_derivative(self, x):
        return 1 if x >= 0 else self.a

    def pw_linear(self, x):
        return x if x >= 0 else self.a * x

    def mask_pw_linear_derivative(self, x):
        mask = x.copy()
        mask[mask >= 0] = 1
        mask[mask <  0] = self.a
        return mask

    def mask_pw_linear(self, x):
        mask = x.copy()
        mask[mask >= 0] = 1
        mask[mask <  0] = self.a
        return mask * x

    def softmax(self, w):
        max = numpy.amax(w, axis=0)
        e = numpy.exp(w - max)
        dist = e / numpy.sum(e, axis=0)
        return dist

    def adjust_a(self, x):
        self.a = x

    def feed_forward(self, activation_input):
        input = activation_input
        iteration_max = self.number_of_layers - 2
        iteration = 0
        for theta in self.thetas:
            if iteration != iteration_max :
                output = self.mask_pw_linear(numpy.dot(theta, input))
            else:
                output = self.softmax(numpy.dot(theta, input))
            input = output
            input = numpy.append(input, 1.0)
            iteration = iteration + 1

        return output
            
    def stochatstic_gradient_descent(self, train_set, valid_set, test_set, minibatch_size, learning_rate):

        # On change l'ordre de nos données au hasard pour les minibatchs
        train_x, train_y = train_set
        converged = False
        train_x = numpy.c_[train_x, numpy.ones((train_x.shape[0], 1))]
        train_set = numpy.c_[train_x, train_y]

        k = 0
        precisions = []
        while not converged:
            numpy.random.shuffle(train_set)
            mini_batches = [train_set[i : i + minibatch_size] for i in range(0, train_set.shape[0], minibatch_size)]

            for mini in mini_batches:
                self.update_with_minibatch(mini, learning_rate, len(mini_batches))

            k = k + 1
            precision = self.precision(test_set)
            print(precision)
            precisions.append(precision)
            if k > 1000:
                converged = True

        numpy.savetxt("1Layers30.csv", precisions, delimiter=",")
        return

    def update_with_minibatch(self, mini, learning_rate, number_of_minibatches):

        # On initialise les poids et les biais
        thetas = [numpy.zeros(theta.shape) for theta in self.thetas]

        # On formatte le mini batch de manière à séparé les x et les y
        data_x = mini[:, 0:-1]
        data_y = mini[:, -1]

        #for x, y in zip(data_x, data_y):
        theta_gradients = self.back_propagation(data_x, data_y) 
        thetas = [theta + theta_gradient / data_x.shape[0] for theta, theta_gradient in zip(thetas, theta_gradients)]

        # On met à jour les biais et les poids
        self.thetas = [theta - learning_rate / 50 * new_theta for theta, new_theta in zip(self.thetas, thetas)]

        return

    def precision(self, data_set):
        x_data, y_data = data_set
        data_set = numpy.c_[x_data, y_data]
        number_of_data = data_set.shape[0]
        results = []

        for data in data_set:
            data_x = data[0:-1]
            data_x = numpy.append(data_x, 1.0)
            data_y = data[-1]
            results.append((numpy.argmax(self.feed_forward(data_x)), data_y))

        return sum([int(x == y) for x, y in results]) / number_of_data

    def back_propagation(self, x, y):
        # On initialise theta
        theta_gradients = [numpy.zeros(theta.shape) for theta in self.thetas]

        # On calcule les vecteurs d'activations
        activations = []
        activations.append(x)
        zs = []

        iteration_max = self.number_of_layers - 2
        iteration = 0
        input = x
        for theta in self.thetas:
            z = numpy.dot(theta, input.T)
            zs.append(z)
            if iteration != iteration_max :
                input = self.mask_pw_linear(z)
            else:
                input = self.softmax(z)
            ones = numpy.ones((1, input.shape[1]))
            input = numpy.concatenate((input, ones), axis=0)
            input = input.T
            activations.append(input)
            iteration = iteration + 1

        delta = activations[-1][:, :-1]
        one_hot_y = numpy.zeros(delta.shape)
        for i, y_value in enumerate(y):
            one_hot_y[i][y_value] = 1

        delta = delta - one_hot_y
        theta_gradients[-1] = numpy.dot(delta.T, activations[-2])

        for i in range(2, self.number_of_layers):
            z = zs[-i]
            derivative = self.mask_pw_linear_derivative(z)
            delta = derivative * numpy.dot(self.thetas[-i + 1].T, delta.T)[:-1]
            theta_gradients[-i] = numpy.dot(delta, activations[-i - 1])
            delta = delta.T

        return theta_gradients