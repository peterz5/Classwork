import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import itertools
import shutil
import tree as tr
from utils import Vocab

class RecursiveNeuralNet:

    def __init__(self, hiddenDim, classes, vocab):
        self.numClasses = classes
        self.hiddenDim = hiddenDim
        self.vocab = vocab

    def initialize_matrices(self, W, b, Ws, bs, L):

        '''
        Initializes the weights and bias matrices, as well as matrices to hold
        their gradients

        args:
            W: initial weight matrix for the composition layer
            b: initial bias for the composition layer
            Ws: initial weight matrix for the projection layer
            bs: initial bias for the projection layer
            L: initial word-embedding matrix
        '''

        self.W = W
        self.b = b
        self.Ws = Ws
        self.bs = bs
        self.L = L  # embedding matrix

        # storing the gradients for the weights and biases

        self.dW = np.zeros((self.hiddenDim, 2*self.hiddenDim), dtype=np.float32)
        self.db = np.zeros((self.hiddenDim, 1), dtype=np.float32)
        self.dWs = np.zeros((self.numClasses, self.hiddenDim), dtype=np.float32)
        self.dbs = np.zeros((self.numClasses, 1), dtype=np.float32)
        self.dL = np.zeros(self.L.shape, dtype=np.float32)

    def forward_prop(self, node):

        '''
        Recursively computes the hidden layer activations for the given node
        during the forward propagation phase. Also computes and updates the
        logits and probabilities for the projection layer at the given node.

        Args:
            node: Node for which to calculate the hidden layer activations and logits
        Returns:
            node.h: hidden layer activation for the composition layer at the given node
        '''

        # base case: leaf nodes
        if node.isLeaf:
            # find index of word in dictionary
            word_id = self.vocab.encode(node.word)

            # select embedding at the given index (L is the embedding matrix)
            node.h = self.L[word_id].T
            # pass through hidden layer and compute softmax
            node_logits = np.dot(self.Ws, node.h) + self.bs
            node.probs = self.softmax(node_logits)

            return node.h

        # calculate hidden layer activations for children
        left_tensor = self.forward_prop(node.left)
        right_tensor = self.forward_prop(node.right)

        # calculate hidden layer activation for current node using those for children
        # multiplying by the appropriate weights matrix and adding the appropriate bias
        node.h = self.relu(np.dot(self.W, np.vstack((left_tensor, right_tensor))) + self.b)

        # compute softmax for classifying current node
        node_logits = np.dot(self.Ws, node.h) + self.bs
        node.probs = self.softmax(node_logits)

        return node.h

    def backward_prop(self, node, errors = None):
        '''
        Computes and updates the gradients for the weights and biases in the
        network arising from a given node, and backpropagates the error to the
        children nodes (if any)

        Args:
            node: Node for which to compute gradients
            errors: Errors (deltas) backpropagated from parent node, if any
        '''

        # Softmax grads
        deltas = node.probs
        deltas[node.label] -= 1.0

        # add gradient from current node for classifier weights and biases
        # i.e., add to self.dWs and self.dbs the respective gradients from this node
        # Hint: You will find the derivatives in the spec helpful

        # YOUR CODE HERE
        self.dWs += deltas * (node.h).T
        self.dbs += deltas
        # END YOUR CODE

        deltas = np.dot(self.Ws.T, deltas)
    
        # add errors from parents (if any) to deltas and
        # backpropagate the deltas through the Relu layer to get deltas_relu

        # YOUR CODE HERE
        if errors != None:
            deltas +=  errors
        matrix = [[1 if node.h[x][y] > 0 else 0 for y in range(len(node.h[0]))] for x in range(len(node.h))]

        deltas_relu = np.matrix(deltas)
        for i in range(len(deltas_relu)):
            for j in range(len(deltas_relu[0])):
                deltas_relu[i][j] *= matrix[i][j]
    
        # END YOUR CODE

        # if node is a leaf
        if node.isLeaf:
            # add deltas_relu to gradient for the current word's embedding
            word_id = self.vocab.encode(node.word)

            new_embedding_grad = self.dL[word_id].reshape((self.hiddenDim, 1)) + deltas_relu
            self.dL[word_id] = new_embedding_grad.T.reshape((self.hiddenDim,))

            return

        if not node.isLeaf:
            # compute and add gradient for the weights and biases for composition
            # arising from current node, i.e., add the respective gradients to
            # self.dW and self.db

            # YOUR CODE HERE
        
            self.dW += np.matmul(deltas_relu, np.vstack((node.left.h, node.right.h)).T)
            self.db += deltas_relu
            # END YOUR CODE

            # compute deltas for children nodes
            deltas = np.dot(self.W.T, deltas_relu)

            # recursively backprop deltas to children node,
            # remember to split the deltas vectors into 2 halves, one for each child

            # YOUR CODE HERE
            a,b = np.split(deltas, 2)

            self.backward_prop(node.left, a)
            self.backward_prop(node.right, b)
            # END YOUR CODE

        return

    def relu(self, X):
        return np.multiply(X, (X > 0.0))

    def softmax(self, X):
        exps = np.exp(X - np.max(X))
        return exps / np.sum(exps)
