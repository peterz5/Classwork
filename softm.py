import os
import numpy as np
import tensorflow as tf

def softmax(x):
  """
  Args:
    x:   tf.Tensor with shape (n_samples, n_features). Note feature vectors are
         represented by row-vectors.
  Returns:
    out: tf.Tensor with shape (n_sample, n_features). You need to construct this
         tensor in this problem.
  """

  ### YOUR CODE HERE

  m = tf.reduce_max(x, reduction_indices = [1], keep_dims = True)
  n = x - m

  y = tf.exp(n)
  sums = tf.reduce_sum(y, reduction_indices = [1], keep_dims = True)
  out = y/sums 
 
  return out
  ### END YOUR CODE

def cross_entropy_loss(y, yhat):
  """

  Args:
    y:    tf.Tensor with shape (n_samples, n_classes). One-hot encoded.
    yhat: tf.Tensorwith shape (n_sample, n_classes). Each row encodes a
          probability distribution and should sum to 1.
  Returns:
    out:  tf.Tensor with shape (1,) (Scalar output). You need to construct this
          tensor in the problem.
  """
  ### YOUR CODE HERE

  z = tf.cast(y, tf.float32)
  out = -tf.reduce_sum(z * tf.log(yhat)) 

  ### END YOUR CODE
  return out


def test_softmax_basic():
  """
  Some simple tests to get you started. 
  Warning: these are not exhaustive.
  """
  print("Running basic tests...")
  test1 = softmax(tf.convert_to_tensor(
      np.array([[1001,1002],[3,4]]), dtype=tf.float32))
  with tf.Session():
      test1 = test1.eval()
  assert np.amax(np.fabs(test1 - np.array(
      [0.26894142,  0.73105858]))) <= 1e-6

  test2 = softmax(tf.convert_to_tensor(
      np.array([[-1001,-1002]]), dtype=tf.float32))
  with tf.Session():
      test2 = test2.eval()
  assert np.amax(np.fabs(test2 - np.array(
      [0.73105858, 0.26894142]))) <= 1e-6

  print("Basic (non-exhaustive) softmax tests pass\n")

def test_cross_entropy_loss_basic():
  """
  Some simple tests to get you started.
  Warning: these are not exhaustive.
  """
  y = np.array([[0, 1], [1, 0], [1, 0]])
  yhat = np.array([[.5, .5], [.5, .5], [.5, .5]])

  test1 = cross_entropy_loss(
      tf.convert_to_tensor(y, dtype=tf.int32),
      tf.convert_to_tensor(yhat, dtype=tf.float32))
  with tf.Session():
    test1 = test1.eval()
  result = -3 * np.log(.5)
  assert np.amax(np.fabs(test1 - result)) <= 1e-6
  print("Basic (non-exhaustive) cross-entropy tests pass\n")

if __name__ == "__main__":
  test_softmax_basic()
  test_cross_entropy_loss_basic()
