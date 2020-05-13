import BPNet
import numpy as np

def predict(X):
    if X >= 0.5:
        return 1
    else:
        return 0


if __name__ == '__main__':
    net = BPNet.BPNet(2, 4, 1)
    X = np.array([[1,0],[1,1],[0,1],[0,0]])
    y = np.array([[1,0,1,0]])
    net.fit(X,y)
    test = [[0,1],[0,0],[1,0],[1,1]]
    true = [1,0,1,0]
    hidden,output = net.forward(test)
    flag = True
    for i in range(4):
        if predict(output[i][0]) != true[i]:
            flag = False
    print(flag,end="")