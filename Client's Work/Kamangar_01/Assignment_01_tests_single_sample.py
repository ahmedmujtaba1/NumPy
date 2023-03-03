import numpy as np
# Modify the line below based on your last name
# for example:
from Kamangar_01_01 import multi_layer_nn
# from Your_last_name_01_01 import multi_layer_nn

def sigmoid(x):
    # This function calculates the sigmoid function
    # x: input
    # return: sigmoid(x)
    # Your code goes here
    return 1/(1+np.exp(-x))

def create_toy_data_nonlinear(n_samples=1000):
    X = np.zeros((n_samples, 4))
    X[:, 0] = np.linspace(-1, 1, n_samples)
    X[:, 1] = np.linspace(-1, 1, n_samples)
    X[:, 2] = np.linspace(-1, 1, n_samples)
    X[:, 3] = np.linspace(-1, 1, n_samples)

    y = X[:, 0]**2 + 2*X[:, 1]  - 0.5*X[:, 2] + X[:, 3]**3 + 0.3

    # shuffle X and y
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    return X.T, y[:, np.newaxis].T

def create_toy_data_nonlinear_2d(n_samples=1000):
    X = np.zeros((n_samples, 4))
    X[:, 0] = np.linspace(-1, 1, n_samples)
    X[:, 1] = np.linspace(-1, 1, n_samples)
    X[:, 2] = np.linspace(-1, 1, n_samples)
    X[:, 3] = np.linspace(-1, 1, n_samples)

    y = np.zeros((n_samples, 2))
    y[:, 0] = 0.5*X[:, 0] -0.2 * X[:, 1]**2 - 0.2*X[:, 2] + X[:, 3]*X[:,1] - 0.1
    y[:, 1] = 1.5 * X[:, 0] + 1.25 * X[:, 1]*X[:, 0] + 0.4 * X[:, 2] * X[:, 0]

    # shuffle X and y
    idx = np.arange(n_samples)
    np.random.shuffle(idx)
    X = X[idx]
    y = y[idx]

    return X.T, y.T


def test_can_fit_data_test_exact():
    np.random.seed(12345)
    X, y = create_toy_data_nonlinear(n_samples=110)
    y = sigmoid(y)
    X_train = X[:, :100]
    X_test = X[:, 100:]
    Y_train = y[:, :100]
    Y_test = y[:, 100:]

    [W, err, Out] = multi_layer_nn(X_train, Y_train, X_test, Y_test, [2, 1], alpha=0.35, epochs=10, h=1e-8,
                                   seed=1234)
    assert abs(err[9] - 0.005176511912601132) < 1e-5


def test_can_fit_data_test_2d():
    np.random.seed(1234)
    X, y = create_toy_data_nonlinear_2d(110)
    y = sigmoid(y)
    X_train = X[:, :100]
    X_test = X[:, 100:]
    Y_train = y[:, :100]
    Y_test = y[:, 100:]

    [W, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,2],alpha=0.35,epochs=10,h=1e-8,seed=1234)
    assert err[1] < err[0]
    assert err[2] < err[1]
    assert err[3] < err[2]
    assert err[9] < 0.005
    assert abs(err[9] - 0.0022172173471326816) < 1e-5



def test_check_weight_init():
    np.random.seed(1234)
    X, y = create_toy_data_nonlinear(n_samples=110)
    X_train = X[:, :100]
    X_test = X[:, 100:]
    Y_train = y[:, :100]
    Y_test = y[:, 100:]

    np.random.seed(1234)
    [W, err, Out] = multi_layer_nn(X_train, Y_train, X_test, Y_test, [2, 1], alpha=0.35, epochs=0, h=1e-8, seed=1234)


    assert np.allclose(W[0], np.array([[0.47143516, -1.19097569, 1.43270697, -0.3126519, -0.72058873],
                                    [0.88716294, 0.85958841, -0.6365235, 0.01569637, -2.24268495]]))
    assert np.allclose(W[1], np.array([[0.47143516, -1.19097569, 1.43270697]]))


def test_large_alpha_test():
    # if alpha is too large, the weights will change too much with each update, and the error will either increase or not improve much

    X, y = create_toy_data_nonlinear(n_samples=110)
    X_train = X[:, :100]
    X_test = X[:, 100:]
    Y_train = y[:, :100]
    Y_test = y[:, 100:]

    [W, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,1],alpha=10,epochs=1,h=1,seed=2)
    assert err[-1] > 0.3


def test_small_alpha_test():
    # if the alpha value is very small (e.g. 1e-9), the weights should not change much with each update, and the error should not decrease
    X, y = create_toy_data_nonlinear(n_samples=110)
    X_train = X[:, :100]
    X_test = X[:, 100:]
    Y_train = y[:, :100]
    Y_test = y[:, 100:]

    [W, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,1],alpha=1e-9,epochs=10,h=1e-8,seed=2)
    assert abs(err[-1] - err[-2]) < 1e-5
    assert abs(err[1] - err[0]) < 1e-5

def test_number_of_nodes_test():
    # check if the number of nodes is being used in creating the weight matrices
    X, y = create_toy_data_nonlinear(n_samples=110)
    X_train = X[:, :100]
    X_test = X[:, 100:]
    Y_train = y[:, :100]
    Y_test = y[:, 100:]

    [W, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[100,1],alpha=1e-9,epochs=0,h=1e-8,seed=2)

    assert W[0].shape == (100, 5)
    assert W[1].shape == (1, 101)

    [W, err, Out] = multi_layer_nn(X_train, Y_train, X_test, Y_test, [42, 1], alpha=1e-9,
                                   epochs=0, h=1e-8, seed=2)
    assert W[0].shape == (42, 5)
    assert W[1].shape == (1, 43)

    X, y = create_toy_data_nonlinear_2d(n_samples=110)
    X_train = X[:, :100]
    X_test = X[:, 100:]
    Y_train = y[:, :100]
    Y_test = y[:, 100:]

    [W, err, Out] = multi_layer_nn(X_train, Y_train, X_test, Y_test, [42, 2], alpha=1e-9,
                                   epochs=0, h=1e-8, seed=2)
    assert W[0].shape == (42, 5)
    assert W[1].shape == (2, 43)

def test_check_output_shape():
    # check if the number of nodes is being used in creating the weight matrices
    X, y = create_toy_data_nonlinear(n_samples=110)
    X_train = X[:, :100]
    X_test = X[:, 100:]
    Y_train = y[:, :100]
    Y_test = y[:, 100:]

    [W, err, Out] = multi_layer_nn(X_train, Y_train, X_test, Y_test, [100, 1], alpha=1e-9, epochs=0, h=1e-8, seed=2)
    assert Out.shape == Y_test.shape

def test_check_output_shape_2d():
    np.random.seed(1234)
    from sklearn.model_selection import train_test_split
    X, y = create_toy_data_nonlinear_2d(110)
    y = sigmoid(y)
    X_train = X[:, :100]
    X_test = X[:, 100:]
    Y_train = y[:, :100]
    Y_test = y[:, 100:]

    [W, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,2],alpha=0.35,epochs=1,h=1e-8,seed=1234)
    assert Out.shape == Y_test.shape

def test_check_output_values():
    np.random.seed(1234)
    X, y = create_toy_data_nonlinear_2d(110)
    y = sigmoid(y)
    X_train = X[:, :100]
    X_test = X[:, 100:]
    Y_train = y[:, :100]
    Y_test = y[:, 100:]

    [W, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,2],alpha=0.35,epochs=0,h=1e-8,seed=1234)
    expected_Out = np.array([[0.70686891, 0.70812892, 0.68737609, 0.71035, 0.70773683,
                           0.64097189, 0.68198946, 0.69904454, 0.62661456, 0.68983985],
                          [0.48811123, 0.48909097, 0.47352803, 0.49085461, 0.48878507,
                           0.44113228, 0.46962086, 0.48215807, 0.43160549, 0.47533035]])
    assert np.allclose(Out, expected_Out, atol=1e-5)

def test_check_weight_update():
    np.random.seed(1234)
    X, y = create_toy_data_nonlinear_2d(110)
    y = sigmoid(y)
    X_train = X[:, :100]
    X_test = X[:, 100:]
    Y_train = y[:, :100]
    Y_test = y[:, 100:]

    np.random.seed(1234)
    [W_before, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,2],alpha=0.2,epochs=0,h=1e-8,seed=1234)
    np.random.seed(1234)
    [W_after, err, Out] = multi_layer_nn(X_train, Y_train, X_test, Y_test, [2, 2], alpha=0.2, epochs=1, h=1e-8, seed=1234)
    delta1 = (W_after[0] - W_before[0])
    delta2 = (W_after[1] - W_before[1])

    correct_delta1 = np.array([[-0.00429947, -0.12501811, -0.12501811, -0.12501811, -0.12501812],
                            [0.05919799, 0.13948422, 0.13948422, 0.13948422, 0.13948422]])
    correct_delta2 = np.array([[-0.23519989, -0.17988833, -0.21721119],
                            [0.37353127, 0.08978976, 0.04998413]])

    assert np.allclose(delta1, correct_delta1, atol=1e-5)
    assert np.allclose(delta2, correct_delta2, atol=1e-5)


def test_h_value_used():
    np.random.seed(1234)
    X, y = create_toy_data_nonlinear_2d(110)
    y = sigmoid(y)
    X_train = X[:, :100]
    X_test = X[:, 100:]
    Y_train = y[:, :100]
    Y_test = y[:, 100:]

    np.random.seed(1234)
    [W_before, err, Out] = multi_layer_nn(X_train,Y_train,X_test,Y_test,[2,2],alpha=0.2,epochs=0,h=1e-8,seed=1234)
    np.random.seed(1234)
    [W_after, err, Out] = multi_layer_nn(X_train, Y_train, X_test, Y_test, [2, 2], alpha=0.2, epochs=1, h=10, seed=1234)
    # if we use some large value for h instead of 1e-8, we should get a different result
    # this will check if the students are using the h value

    assert not np.allclose(W_after[0], W_before[0], atol=1e-5)
    assert not np.allclose(W_after[1], W_before[1], atol=1e-5)
