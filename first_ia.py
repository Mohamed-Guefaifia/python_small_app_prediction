import numpy as np


x_entree = np.array(([3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[2,0.5],[5.5,1],[1,1],[4,1.5]),dtype=float)
y = np.array(([1],[0],[1],[0],[1],[0],[1],[0]), dtype=float)   # Donnees de sortie 1 = rouge / 0 = bleu

x_entree = x_entree / np.amax(x_entree, axis=0)
#print (x_entree)

X = np.split(x_entree, [8])[0]
x_prediction = np.split(x_entree,[8])[1]

#print (x_prediction)

class Neural_Network(object):
    def __init__(self):
        self.InputSize = 2
        self.OutputSize = 1
        self.HiddenSize = 3

        self.w1 = np.random.randn(self.InputSize, self.HiddenSize) # Matrice 2*3
        self.w2 = np.random.randn(self.HiddenSize , self.OutputSize) # Matrice 3*1
    
    def forward(self, X):
        self.z = np.dot(X, self.w1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.w2)
        out = self.sigmoid(self.z3)
        return out

    def sigmoid(self,s):
        return 1/(1 + np.exp(-s))
    

    def sigmoidPrime(self, s):
        return s * (1 - s)
    

    def backward(self,X, y, o):

       self.o_error = y - 0
       self.o_delta =  self.o_error * self.sigmoidPrime(o)
       self.z2_error = self.o_delta.dot(self.w2.T)
       self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

       self.w1 += X.T.dot(self.z2_delta)
       self.w2 += self.z2.T.dot(self.o_delta)

    
    def train(self , X,y):
        out = self.forward(X)
        self.backward(X,y ,out)







NN = Neural_Network()

for i in range(300):
    print("# " + str(i) + "\n" )
    print("Valeur d'entree : \n" + str(X))
    print("Sortie actuelle : \n" + str(y))
    print("Sortie predite : \n" + str(np.matrix.round(NN.forward(X),2)))
    print("\n")
    NN.train(X,y)


#print ("Sortie predit par l'IA : \n" + str(out))
#print ("Vrai Sortie : \n" + str(y))