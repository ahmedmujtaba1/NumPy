import numpy as np

def multi_layer_nn(X_train, Y_train, X_test, Y_test, layers, alpha, epochs, h=0.00001, seed=2, num_classes=1):
    # initialize weight matrices for all layers
    Ws = []
    np.random.seed(seed)
    num_features = X_train.shape[0]
    num_train_samples = X_train.shape[1]

    layer_dims = layers
    # print(layer_dims)
    for i in range(len(layer_dims)):
        if i == 0:
            W = np.random.randn(layer_dims[i], num_features+1)
        else:
            W = np.random.randn(layer_dims[i], layer_dims[i-1]+1)
        Ws.append(W)

    # add bias term to the input data
    X_train = np.insert(X_train, 0, 1, axis=0)
    X_test = np.insert(X_test, 0, 1, axis=0)

    # initialize variables for storing error and predicted outputs
    errors = []
    predicted_outputs = []

    for epoch in range(epochs):
        # forward pass
        A = [X_train] # list of activation matrices
        Z = [] # list of linear combinations of weight and activation matrices

        for i in range(len(layer_dims)):
            Z_i = np.dot(Ws[i], A[i])
            Z.append(Z_i)
            if i == len(layer_dims) - 1:
                A_i = sigmoid(Z_i)
            else:
                A_i = np.insert(sigmoid(Z_i), 0, 1, axis=0)
            A.append(A_i)

        # calculate error
        if num_classes == 1:
            error = np.sum((A[-1] - Y_train) ** 2) / num_train_samples
        else:
            error = np.sum((A[-1] - Y_train.T) ** 2) / num_train_samples
        errors.append(error)

        
        # calculate gradient
        
        # calculate gradient
        dWs = [None] * len(layer_dims)
        
        for i in range(len(layer_dims) - 1, -1, -1):
            if i == len(layer_dims) - 1:
                delta_i = (A[-1] - Y_train) * A[-1] * (1 - A[-1])
            else:
                if i == 0:
                    delta_i = np.dot(Ws[i+1].T[1:, :], delta_i) * A[i] * (1 - A[i])
                    # delta_i = np.dot(Ws[i+1].T[1:, :], delta_i) * A[i+1] * (1 - A[i+1])
                else:
                    # delta_i = np.dot(Ws[i+1].T[1:, :], delta_i)[1:,:] * A[i] * (1 - A[i])[1:,:

                    delta_i = np.dot(Ws[i+1].T[1:, :], delta_i) * A[i][1:, :] * (1 - A[i][1:, :])
            dW_i = np.dot(delta_i, A[i].T) / num_train_samples
            dWs[i] = dW_i
            # print(dWs)

        # update weights
        for i in range(len(layer_dims)):
            Ws[i] = Ws[i] - alpha * dWs[i]
    # evaluate on test data
    A = [X_test]
    for i in range(len(layer_dims)):
        Z_i = np.dot(Ws[i], A[i])
        if i == len(layer_dims) - 1:
            A_i = sigmoid(Z_i)
        else:
            A_i = np.insert(sigmoid(Z_i), 0, 1, axis=0)
        A.append(A_i)
    predicted_outputs = A[-1]

    return Ws, errors, predicted_outputs

def sigmoid(x):
    return 1/(1+np.exp(-x))


def create_toy_data_nonlinear(n_samples=1000):
    X = np.zeros((4, n_samples))
    X[0, :] = np.linspace(-1, 1, n_samples)
    X[1, :] = np.linspace(-1, 1, n_samples)
    X[2, :] = X[0, :]**2
    X[3, :] = X[1, :]**2
    Y = (np.sin(X[0, :] + X[1, :]) + np.cos(X[2, :] + X[3, :]))/2
    return X, Y


if __name__ == "__main__":
    # generate toy non-linear data
    X_train, Y_train = create_toy_data_nonlinear()
    X_test, Y_test = create_toy_data_nonlinear()
    
    # set layer dimensions
    layers = [4, 4, 1]
    
    # set the learning parameters
    alpha = 0.001
    epochs = 10000
    
    # train the model
    Ws, errors, predicted_outputs = multi_layer_nn(X_train, Y_train, X_test, Y_test, layers, alpha, epochs, num_classes=1)
    
    # calculate the accuracy
    accuracy = np.mean(predicted_outputs == Y_test)
    print("Accuracy:", accuracy)