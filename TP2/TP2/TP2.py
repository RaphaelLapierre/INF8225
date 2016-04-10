import random
import matplotlib.pyplot as plt
import numpy as np
import csv

def main():

    # Read inputs
    with open('./documents.txt', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        X = []
        for row in reader:
            X.append([int(element) for element in row])

    with open('./groupnames.txt', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            groupnames = row

    with open('./newsgroups.txt', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            j = [int(element) for element in row]

    with open('./wordlist.txt', 'rt') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            wordlist = row

    # Format inputs
    X.append([1 for i in range (len(j))])

    X = np.array(X)
    X = X.T

    j = np.array(j)

    n = 4
    m = len(j)

    percent = [int(0.7 * m) + 1, int(0.85 * m) + 1]

    Y = []

    for i in range(len(j)):
        Y.append([0, 0, 0, 0])
        Y[i][j[i] - 1] = 1

    Y = np.array(Y)

    ZipYX = np.c_[Y, X]
    np.random.shuffle(ZipYX);
    Y, X = np.hsplit(ZipYX, [4]);

    YA, YV, YT = np.vsplit(Y, percent)
    XA, XV, XT = np.vsplit(X, percent)

    Theta = []
    for i in range(4):
        Theta.append([random.random() - 0.5 for i in range(101)])
            
    Theta = np.array(Theta)

    learning_rate = 0.0005

    # Approche par batch
    print("Batch")
    batch_precisionA_output = []
    batch_precisionV_output = []
    batch_log_vraisemblance_output = []

    converged = True
    Gauche = np.dot(YA.T, XA)
    while not converged:
        Y_possibility = np.identity(n)
        Z = np.sum(np.exp(np.dot(Y_possibility, np.dot(Theta, XV.T))), 0);
        LogVraisemblance = np.sum(np.dot(YV, Theta) * XV, 1) - np.log(Z)
        LogVraisemblance = np.sum(LogVraisemblance)
        batch_log_vraisemblance_output.append(LogVraisemblance)

        Z = np.sum(np.exp(np.dot(Y_possibility, np.dot(Theta, XA.T))), 0);
        Z = np.c_[Z, Z, Z, Z]
        Droite = np.dot((np.exp(np.dot(Y_possibility, np.dot(Theta, XA.T))).T / Z).T, XA)
        Gradient = Gauche - Droite
        Theta = Theta + learning_rate * Gradient

        # Precision
        batch_precisionA_output.append(precision(XA, YA, Theta, n))
        batch_precisionV_output.append(precision(XV, YV, Theta, n))

        if len(batch_log_vraisemblance_output) > 1:
            converged = (batch_log_vraisemblance_output[-1] - batch_log_vraisemblance_output[-2]) < 0.5

    plt.plot(batch_log_vraisemblance_output)
    plt.title("Log vraisemblance de la méthode sans batch")
    plt.ylabel("Log vraisemblance")
    plt.xlabel("Iteration")
    plt.show()

    plt.plot(batch_precisionV_output, label="Validation")
    plt.plot(batch_precisionA_output, label="Apprentissage")
    plt.title("Graphique de la précision sur les ensembles d'apprentissage et de validation")
    plt.ylabel("Précision")
    plt.xlabel("Itération")
    plt.legend(loc="lower right")
    plt.show()
    # Approche par minibatch
    print("Mini Batch")

    Theta = []
    for i in range(4):
        Theta.append([random.random() - 0.5 for i in range(101)])
    Theta = np.array(Theta)
    NumberOfBatches = 20
    Alpha = 0.5
    DeltaTheta = np.zeros((4, 101))
    converged = True
    iteration = -1
    learning_rate = 0.0005

    minibatch_precisionA_output = []
    minibatch_precisionV_output = []
    minibatch_log_vraisemblance_output = []

    while not converged:
        iteration = iteration + 1
        listXB, listYB = create_mini_batch(XA, YA, 20)

        for XB, YB in zip(listXB, listYB):
            Y_possibility = np.identity(n)
            Z = np.sum(np.exp(np.dot(Y_possibility, np.dot(Theta, XV.T))), 0);
            LogVraisemblance = np.sum(np.dot(YV, Theta) * XV, 1) - np.log(Z)
            LogVraisemblance = np.sum(LogVraisemblance)
            minibatch_log_vraisemblance_output.append(LogVraisemblance)
            
            Z = np.sum(np.exp(np.dot(Y_possibility, np.dot(Theta, XB.T))), 0);
            Z = np.c_[Z, Z, Z, Z]

            Gauche = np.dot(YB.T, XB)
            Droite = np.dot((np.exp(np.dot(Y_possibility, np.dot(Theta, XB.T))).T / Z).T, XB)

            Gradient = (Gauche - Droite) / NumberOfBatches
            DeltaTheta = Alpha * DeltaTheta + learning_rate * Gradient
            Theta = Theta + DeltaTheta

            minibatch_precisionA_output.append(precision(XA, YA, Theta, n))
            minibatch_precisionV_output.append(precision(XV, YV, Theta, n))

            if len(minibatch_log_vraisemblance_output) > 1:
                converged = (minibatch_log_vraisemblance_output[-1] - minibatch_log_vraisemblance_output[-2]) < 0.5

    plt.plot(minibatch_log_vraisemblance_output)
    plt.title("Log vraisemblance de la méthode de mini-batch sans régularisation")
    plt.ylabel("Log vraisemblance")
    plt.xlabel("Iteration")
    plt.show()

    plt.plot(minibatch_precisionA_output, label="Validation")
    plt.plot(minibatch_precisionV_output, label="Apprentissage")
    plt.title("Graphique de la précision sur les ensembles d'apprentissage et de validation")
    plt.ylabel("Précision")
    plt.xlabel("Itération")
    plt.legend(loc="lower right")
    plt.show()
    # Mini batch avec régularisation
    print("Mini batch with reg")

    # Adding random values to X
    regularizationValues = np.random.randint(2, size=(X.shape[0], 100))
    X = np.append(X, regularizationValues, axis=1)
    XA, XV, XT = np.vsplit(X, percent)

    Theta = []
    for i in range(4):
        Theta.append([random.random() - 0.5 for i in range(201)])
    Theta = np.array(Theta)
    NumberOfBatches = 20
    Alpha = 0.5
    DeltaTheta = np.zeros((4, 201))
    converged = False
    iteration = -1
    learning_rate = 0.0005

    lambda1 = 0.01
    lambda2 = 0.05

    regularization_minibatch_precisionA_output = []
    regularization_minibatch_precisionV_output = []
    regularization_minibatch_log_vraisemblance_output = []


    while not converged:
        iteration = iteration + 1
        listXB, listYB = create_mini_batch(XA, YA, 20)

        for XB, YB in zip(listXB, listYB):
            learningSize = XA.shape[0]
            batchSize = XB.shape[0]
            Y_possibility = np.identity(n)
            Z = np.sum(np.exp(np.dot(Y_possibility, np.dot(Theta, XV.T))), 0);
            LogVraisemblance = np.sum(np.dot(YV, Theta) * XV, 1) - np.log(Z)
            LogVraisemblance = np.sum(LogVraisemblance)
            regularization_minibatch_log_vraisemblance_output.append(LogVraisemblance)
            
            Z = np.sum(np.exp(np.dot(Y_possibility, np.dot(Theta, XB.T))), 0);
            Z = np.c_[Z, Z, Z, Z]

            Gauche = np.dot(YB.T, XB)
            Droite = np.dot((np.exp(np.dot(Y_possibility, np.dot(Theta, XB.T))).T / Z).T, XB)

            Gradient = (Gauche - Droite) / NumberOfBatches + batchSize /learningSize * (lambda2 * 2 * Theta + lambda1 * ((Theta > 0) + (Theta < 0) * -1))
            DeltaTheta = Alpha * DeltaTheta + learning_rate * Gradient
            Theta = Theta + DeltaTheta

            regularization_minibatch_precisionA_output.append(precision(XA, YA, Theta, n))
            regularization_minibatch_precisionV_output.append(precision(XV, YV, Theta, n))

            if len(regularization_minibatch_log_vraisemblance_output) > 1:
                converged = (regularization_minibatch_log_vraisemblance_output[-1] - regularization_minibatch_log_vraisemblance_output[-2]) < 0.5

    plt.plot(regularization_minibatch_log_vraisemblance_output)
    plt.title("Log vraisemblance de la méthode de mini-batch avec régularisation")
    plt.ylabel("Log vraisemblance")
    plt.xlabel("Iteration")
    plt.show()

    plt.plot(regularization_minibatch_precisionA_output, label="Validation")
    plt.plot(regularization_minibatch_precisionV_output, label="Apprentissage")
    plt.title("Graphique de la précision sur les ensembles d'apprentissage et de validation")
    plt.ylabel("Précision")
    plt.xlabel("Itération")
    plt.legend(loc="lower right")
    plt.show()

    plt.hist(Theta.reshape((4*201, 1)))
    plt.title("Histogramme de Theta")
    plt.ylabel("Valeur")
    plt.show()

    print("All converged")


def precision(X, Y, Theta, n):
    Z = np.sum(np.exp(np.dot(np.identity(n), np.dot(Theta, X.T))), 0);
    Z = np.c_[Z, Z, Z, Z]

    probability = np.exp(np.dot(np.identity(n), np.dot(Theta, X.T))) / Z.T
    prediction = np.argmax(probability, axis=0)
    values = np.argmax(Y, axis=1)
    return sum([prediction[i] == values[i] for i in range(prediction.size)]) / prediction.size

def create_mini_batch(X, Y, number_of_batches):
    XB = []
    YB = []
    ZipYX = np.c_[Y, X]
    np.random.shuffle(ZipYX);
    ZipYX = np.array_split(ZipYX, number_of_batches)
    for batch in ZipYX:
        Y, X = np.hsplit(batch, [4]);
        YB.append(Y)
        XB.append(X)

    return XB, YB

if __name__ == "__main__":
    main()
