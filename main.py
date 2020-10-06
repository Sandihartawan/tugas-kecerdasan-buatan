import numpy as np
np.random.seed(10)
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

data = pd.read_csv('./train.csv')

data.head(4)

# We define a dictionary to transform the 0,1 values in the labels to a String that defines the fate of the passenger
dict_live = {
    0: 'Perished',
    1: 'Survived'
}

# We define a dictionary to binarize the sex
dict_sex = {
    'male': 0,
    'female': 1
}

# We apply the dictionary using a lambda function and the pandas .apply() module
data['Bsex'] = data['Sex'].apply(lambda x: dict_sex[x])

# Now the features are a 2 column matrix whose entries are the Class (1,2,3) and the Sex (0,1) of the passengers
features = data[['Pclass', 'Bsex']].to_numpy()
labels = data['Survived'].to_numpy()

# Define the sigmoid activator; we ask if we want the sigmoid or its derivative
def sigmoid_act(x, der=False):
    if (der == True):  # derivative of the sigmoid
        f = 1 / (1 + np.exp(- x)) * (1 - 1 / (1 + np.exp(- x)))
    else:  # sigmoid
        f = 1 / (1 + np.exp(- x))

    return f


# We may employ the Rectifier Linear Unit (ReLU)
def ReLU_act(x, der=False):
    if (der == True):  # the derivative of the ReLU is the Heaviside Theta
        f = np.heaviside(x, 1)
    else:
        f = np.maximum(x, 0)

    return f


def run():
    X_train, X_test, Y_train, Y_test = train_test_split(features, labels, test_size=0.35)
    print('Training records:', Y_train.size)
    print('Test records:', Y_test.size)

    # Set up the number of perceptron per each layer:
    p = 4  # Layer 1
    q = 4  # Layer 2

    # Set up the Learning rate
    eta = 1 / 623

    # 0: Random initialize the relevant data
    w1 = 2 * np.random.rand(p, X_train.shape[1]) - 0.5  # Layer 1
    b1 = np.random.rand(p)

    w2 = 2 * np.random.rand(q, p) - 0.5  # Layer 2
    b2 = np.random.rand(q)

    wOut = 2 * np.random.rand(q) - 0.5  # Output Layer
    bOut = np.random.rand(1)

    mu = []
    vec_y = []

    # Start looping over the passengers, i.e. over I.

    for I in range(0, X_train.shape[0]):  # loop in all the passengers:

        # 1: input the data
        x = X_train[I]

        # 2.1: Feed forward
        z1 = ReLU_act(np.dot(w1, x) + b1)  # output layer 1
        z2 = ReLU_act(np.dot(w2, z1) + b2)  # output layer 2
        y = sigmoid_act(np.dot(wOut, z2) + bOut)  # Output of the Output layer

        # 2.2: Compute the output layer's error
        delta_Out = (y - Y_train[I]) * sigmoid_act(y, der=True)

        # 2.3: Backpropagate
        delta_2 = delta_Out * wOut * ReLU_act(z2, der=True)  # Second Layer Error
        delta_1 = np.dot(delta_2, w2) * ReLU_act(z1, der=True)  # First Layer Error

        # 3: Gradient descent
        wOut = wOut - eta * delta_Out * z2  # Outer Layer
        bOut = bOut - eta * delta_Out

        w2 = w2 - eta * np.kron(delta_2, z1).reshape(q, p)  # Hidden Layer 2
        b2 = b2 - eta * delta_2

        w1 = w1 - eta * np.kron(delta_1, x).reshape(p, x.shape[0])  # Hidden Layer 1
        b1 = b1 - eta * delta_1

        # 4. Computation of the loss function
        mu.append((1 / 2) * (y - Y_train[I]) ** 2)
        vec_y.append(y[0])

    # Plotting the Cost function for each training data
    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(0, X_train.shape[0]), mu, alpha=0.3, s=4, label='mu')
    plt.title('Loss for each training data point', fontsize=20)
    plt.xlabel('Training data', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.show()

    # Plotting the average cost function over 10 training data
    pino = []
    for i in range(0, 9):
        pippo = 0
        for m in range(0, 59):
            pippo += vec_y[60 * i + m] / 60
        pino.append(pippo)

    plt.figure(figsize=(10, 6))
    plt.scatter(np.arange(0, 9), pino, alpha=1, s=10, label='error')
    plt.title('Averege Loss by epoch', fontsize=20)
    plt.xlabel('Epoch', fontsize=16)
    plt.ylabel('Loss', fontsize=16)
    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    run()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
