import random
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

        if len(batch_log_vraisemblance_output) != 1:
            converged = (batch_log_vraisemblance_output[-1] - batch_log_vraisemblance_output[-2]) < 1.0

    # Approche par minibatch

    Theta = []
    for i in range(4):
        Theta.append([random.random() - 0.5 for i in range(101)])
    Theta = np.array(Theta)
    NumberOfBatches = 20
    Alpha = 0.5
    DeltaTheta = np.zeros((4, 101))
    converged = False
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
