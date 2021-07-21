import numpy as np
import matplotlib.pyplot as plt
import sys

input_dim = 2
output_dim = 1
hidden_layer_dim = 10
num_epoch = 10000
learning_rate = 0.1

np.random.seed(1)

def generate_linear(n=100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs = []
    labels = []
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0]-pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)

def generate_XOR_easy():
    inputs = []
    labels = []
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)

        if 0.1*i == 0.5:
            continue
        inputs.append([0.1*i, 1-0.1*i])
        labels.append(1)

    return np.array(inputs), np.array(labels).reshape(21, 1)

def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title('Ground truth', fontsize = 18)
    for i in range(x.shape[0]):
        if y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    
    plt.subplot(1, 2, 2)
    plt.title('Predict result', fontsize = 18)
    for i in range(x.shape[0]):
        if pred_y[i] == 0:
            plt.plot(x[i][0], x[i][1], 'ro')
        else:
            plt.plot(x[i][0], x[i][1], 'bo')
    plt.show()
    
def show_curve(epoch, loss):
    plt.plot(epoch, loss)
    plt.title("Learning curve")
    plt.ylim((0, 25))
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

def derivative_sigmoid(x):
    return np.multiply(x, 1.0 - x)
  
def loss_function(y, y_hat):
    return np.mean(np.power((y - y_hat), 2))

def forward(weight1,weight2,weight3,x):
    a_list = []

    a = sigmoid(np.matmul(x, weight1))
    a_list.append(a)
    a = sigmoid(np.matmul(a, weight2))
    a_list.append(a)
    a = sigmoid(np.matmul(a, weight3))
    a_list.append(a)

    return a, a_list

def backpropagation(x, y, outputs, weight1, weight2, weight3):
    ## output layer
    a2_z2 = derivative_sigmoid(outputs[2])
    c_a2 = 2 * (outputs[2] - y)
    c_z2 = np.matmul(a2_z2, c_a2)
    c_w2 = np.matmul(outputs[1].T, c_z2)
    
    ## layer 2
    for i in range(len(weight3)):
        a1_z1 = np.array([derivative_sigmoid(outputs[1]).reshape(-1)[i]]).reshape(1, 1)
        z2_a1 = np.array([weight3[i]])
    
        c_a1 = np.matmul(z2_a1, c_z2)
        z1_w1 = outputs[0].T
        c_z1_notconcat = np.matmul(a1_z1, c_a1)
        
        if i == 0:
            c_z1 = np.matmul(a1_z1, c_a1)
            c_w1 = np.matmul(z1_w1, c_z1)
        else:
            c_z1 = np.concatenate((c_z1, np.matmul(a1_z1, c_a1)), axis = 1)
            c_w1 = np.concatenate((c_w1, np.matmul(z1_w1, c_z1_notconcat)), axis = 1)

    ## layer 1
    for i in range(len(weight2)):
        a0_z0 = np.array([derivative_sigmoid(outputs[0]).reshape(-1)[i]]).reshape(1, 1)
        z1_a0 = np.array([weight2[i]]).reshape(len(weight2), 1)

        c_a0 = np.matmul(z1_a0, c_z1)
        z0_w0 = x.T
        c_a0 = np.matmul(z1_a0.T, c_z1.T)
        c_z0 = np.matmul(a0_z0, c_a0)
        if i == 0:
            c_w0 = np.matmul(z0_w0, c_z0)
        else:
            c_w0 = np.concatenate((c_w0, np.matmul(z0_w0, c_z0)), axis = 1)
    
    weight1 -= learning_rate * c_w0
    weight2 -= learning_rate * c_w1
    weight3 -= learning_rate * c_w2
    return weight1, weight2, weight3

def train(x,y):
    ## init network weights
    weight1 = np.random.random((input_dim, hidden_layer_dim))
    weight2 = np.random.random((hidden_layer_dim, hidden_layer_dim))
    weight3 = np.random.random((hidden_layer_dim, output_dim))
    epoch_list = [] # for plot
    loss_list = [] # for plot
    
    ## for each training example
    for i in range(num_epoch):
        loss = 0
        for j in range(len(x)):
            ## forward pass
            pred, outputs = forward(weight1, weight2, weight3, x[j].reshape(1, 2))
            ## loss
            loss += loss_function(pred, y[j].reshape(1, 1))
            ## backpropagation
            weight1, weight2, weight3 = backpropagation(x[j].reshape(1, 2),y[j].reshape(1, 1), outputs, weight1,weight2,weight3)
        epoch_list.append(i)
        loss_list.append(loss)
        if (i+1) % 100 == 0:
            print("epoch {} loss : {}".format(i+1, loss))

    pred, _ = forward(weight1, weight2, weight3, x)
    print("Prediction: ", pred)
    
    num_correct = 0
    for i in range(len(pred)):
        if pred[i] > 0.5:
            pred[i] = 1
        else:
            pred[i] = 0
        if pred[i]==y[i]:
            num_correct +=1
    print("Accuracy: ", num_correct / len(pred))
    
    show_result(x, y, pred)
    show_curve(epoch_list, loss_list)

    return weight1, weight2, weight3

if __name__ == '__main__':
    x1, y1 = generate_linear(n = 100)
    x2, y2 = generate_XOR_easy()
    
    if sys.argv[1] == 'linear':
        w1, w2, w3 = train(x1, y1)
    else:
        w1, w2, w3 = train(x2, y2)
