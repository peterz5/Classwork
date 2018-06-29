import time
import os
import math
import numpy as np
import tensorflow as tf
from q1_softmax import softmax
from q1_softmax import cross_entropy_loss
from model import Model
from utils import data_iterator

class Config(object):
  """Holds model hyperparams and data information.

  The config class is used to store various hyperparameters and dataset
  information parameters. Model objects are passed a Config() object at
  instantiation.
  """
  batch_size = 64
  n_samples = 1024
  n_features = 100
  n_classes = 5
  # You may adjust the max_epochs to ensure convergence.
  max_epochs = 50
  # You may adjust this learning rate to ensure convergence.
  lr = 1e-4 

class SoftmaxModel(Model):
  """Implements a Softmax classifier with cross-entropy loss."""

  def load_data(self):
    """Creates a synthetic dataset and stores it in memory."""
    np.random.seed(1234)
    self.input_data = np.random.rand(self.config.n_samples, self.config.n_features)
    self.input_labels = np.ones((self.config.n_samples,), dtype=np.int32)

  def add_placeholders(self):
    
    ### YOUR CODE HERE
    self.input_placeholder = tf.placeholder(tf.float32, shape=(self.config.batch_size, self.config.n_features),name='input')
    self.labels_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.config.n_classes), name='labels')
    ### END YOUR CODE

  def create_feed_dict(self, input_batch, label_batch):
    
    ### YOUR CODE HERE
    feed_dict = {self.input_placeholder: input_batch, self.labels_placeholder: label_batch }
    ### END YOUR CODE
    return feed_dict

  def add_training_op(self, loss):
  
    ### YOUR CODE HERE
    train_op = tf.train.GradientDescentOptimizer(self.config.lr).minimize(loss)
    ### END YOUR CODE
    return train_op

  def add_model(self, input_data):
    
    ### YOUR CODE HERE
    with tf.variable_scope('softmax-regression'):
        weights = tf.Variable(tf.zeros((self.config.n_features, self.config.n_classes)), name='weights')
        bias = tf.Variable(tf.zeros(self.config.n_classes, name='bias'))
        out = softmax(tf.matmul(input_data, weights) + bias)
    ### END YOUR CODE
    return out
 
  def add_loss_op(self, pred):
    
    ### YOUR CODE HERE
    loss = cross_entropy_loss(self.labels_placeholder, pred)
    ### END YOUR CODE
    return loss

  def run_epoch(self, sess, input_data, input_labels):
    
    # And then after everything is built, start the training loop.
    average_loss = 0
    for step, (input_batch, label_batch) in enumerate(data_iterator(input_data, input_labels,batch_size=self.config.batch_size, label_size=self.config.n_classes)):

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = self.create_feed_dict(input_batch, label_batch)

      # Run one step of the model.  The return values are the activations
      # from the `self.train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([self.train_op, self.loss], feed_dict=feed_dict)
      average_loss += loss_value

    average_loss = average_loss / step
    return average_loss 

  def fit(self, sess, input_data, input_labels):
    """Fit model on provided data.

    Args:
      sess: tf.Session()
      input_data: np.ndarray of shape (n_samples, n_features)
      input_labels: np.ndarray of shape (n_samples, n_classes)
    Returns:
      losses: list of loss per epoch
    """
    losses = []
    for epoch in range(self.config.max_epochs):
      start_time = time.time()
      average_loss = self.run_epoch(sess, input_data, input_labels)
      duration = time.time() - start_time
      # Print status to stdout.
      print('Epoch %d: loss = %.2f (%.3f sec)'
             % (epoch, average_loss, duration))
      losses.append(average_loss)
    return losses

  def __init__(self, config):
    """Initializes the model.

    Args:
      config: A model configuration object of type Config
    """
    self.config = config
    # Generate placeholders for the images and labels.
    self.load_data()
    self.add_placeholders()
    self.pred = self.add_model(self.input_placeholder)
    self.loss = self.add_loss_op(self.pred)
    self.train_op = self.add_training_op(self.loss)
  
def test_SoftmaxModel():
  """Train softmax model for a number of steps."""
  config = Config()
  with tf.Graph().as_default():
    model = SoftmaxModel(config)
  
    # Create a session for running Ops on the Graph.
    sess = tf.Session()
  
    # Run the Op to initialize the variables.
    init = tf.global_variables_initializer()
    sess.run(init)
  
    losses = model.fit(sess, model.input_data, model.input_labels)

  # If ops are implemented correctly, the average loss should fall close to zero
  # rapidly.
  assert losses[-1] < .5
  print("Basic (non-exhaustive) classifier tests pass\n")

if __name__ == "__main__":
    test_SoftmaxModel()
