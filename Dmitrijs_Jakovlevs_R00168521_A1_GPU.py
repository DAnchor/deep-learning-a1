'''
Python: 3.10.xx
GPU Version: DirectML plugin, Batch Size: 2048

REFERENCES:
Deep Learning Tutorial: https://www.geeksforgeeks.org/deep-learning/deep-learning-tutorial/
Introduction to Deep Learning: https://www.geeksforgeeks.org/introduction-deep-learning/
Neural Networks Guide: https://www.geeksforgeeks.org/deep-learning/neural-networks-a-beginners-guide/
Backpropagation: https://www.geeksforgeeks.org/backpropagation-in-neural-network/
Activation Functions: https://www.geeksforgeeks.org/activation-functions-neural-networks/
Categorical Cross-Entropy: https://www.geeksforgeeks.org/deep-learning/categorical-cross-entropy-in-multi-class-classification/
Adam Optimizer: https://www.geeksforgeeks.org/deep-learning/adam-optimizer/
Keras to_categorical: https://www.geeksforgeeks.org/machine-learning/python-keras-keras-utils-to_categorical/
Data Preprocessing: https://www.geeksforgeeks.org/data-preprocessing-machine-learning-python/
Evaluation Metrics: https://www.geeksforgeeks.org/machine-learning/metrics-for-machine-learning-model/
Fashion MNIST: https://www.tensorflow.org/api_docs/python/tf/keras/datasets/fashion_mnist/load_data
'''

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import asyncio as task
from time import perf_counter
from keras.utils import to_categorical
from dataclasses import dataclass


# DATA STRUCTURE AND BENCHMARKING
@dataclass
class Dataset:
    training_input_data: any
    training_output_labels: any
    validation_input_data: any
    validation_output_labels: any
    testing_input_data: any
    testing_output_labels: any


# UTILITY TO MEASURE RUNTIMES
class Benchmark:
    def __init__(self):
        self.benchmark_function_name = ''
        self.benchmark_start_time = 0

    def benchmark_start(self, name):
        self.benchmark_function_name = name
        self.benchmark_start_time = perf_counter()                                              # timer start

    def benchmark_stop(self):
        elapsed_time = perf_counter() - self.benchmark_start_time                               # capture time and compute
        
        print(f'\n>>>\t{self.benchmark_function_name} >>> EXECUTION TIME : {elapsed_time:.6f} SEC\t<<<\n')


benchmark_utility = Benchmark()


# MATH AND UTILS
def forward_pass(input_features, weights, bias):
    # https://www.geeksforgeeks.org/deep-learning/neural-networks-a-beginners-guide/
    return tf.matmul(weights, tf.cast(input_features, tf.float32)) + bias                       # Z = WX + b

def cross_entropy(predicted_probabilities, true_labels):
    # https://www.geeksforgeeks.org/deep-learning/categorical-cross-entropy-in-multi-class-classification/
    sample_count_m = tf.cast(tf.shape(true_labels)[1], tf.float32)                              # sample count M
    numerical_stability_epsilon = 1e-7                                                          # numeric stability
    
    return (-1 / sample_count_m) * tf.reduce_sum(tf.cast(true_labels, tf.float32) * tf.math.log(predicted_probabilities + numerical_stability_epsilon))

def calculate_accuracy(predicted_probabilities, true_labels):
    # https://www.geeksforgeeks.org/machine-learning/metrics-for-machine-learning-model/
    predicted_classes = tf.argmax(predicted_probabilities, axis = 0)                            # get max index
    true_classes = tf.argmax(true_labels, axis = 0)                                             # get index
    correct_predictions = tf.cast(tf.equal(predicted_classes, true_classes), tf.float32)         # boolean > float
    
    return tf.reduce_sum(correct_predictions) / tf.cast(tf.shape(true_labels)[1], tf.float32)


# TRANSPOSE AND SCALER (NORMALISATION AND CATEGORICAL)
async def data_pipeline():
    benchmark_utility.benchmark_start("data_pipeline")                                      # timer start
    
    # https://www.tensorflow.org/datasets/keras_example
    fashion_mnist = tf.keras.datasets.fashion_mnist

    (tr_x, tr_y), (te_x, te_y) = fashion_mnist.load_data()                                  # load the training and test data

    tr_x = tr_x.reshape(tr_x.shape[0], 784)                                                 # reshape the feature data
    te_x = te_x.reshape(te_x.shape[0], 784)

    # https://www.geeksforgeeks.org/data-preprocessing-machine-learning-python/             # noramlise feature data
    tr_x = (tr_x / 255.0).astype(np.float32)
    te_x = (te_x / 255.0).astype(np.float32)

    # https://www.geeksforgeeks.org/machine-learning/python-keras-keras-utils-to_categorical/
    tr_y = to_categorical(tr_y,10)                                                          # encode the test labels for transpose
    tr_y = tr_y.T.astype(np.float32)

    te_y = to_categorical(te_y,10)                                                          
    te_y = te_y.T.astype(np.float32)

    tr_x_transposed = tr_x.T                                                                # transpose matrix multiplication
    te_x_transposed = te_x.T

    # 80 / 20 standard split ratio
    validation_input_data = tr_x_transposed[:, 48000:]                                      # validation split x
    validation_output_labels = tr_y[:, 48000:]                                              # validation split y
    training_input_data = tr_x_transposed[:, :48000]                                        # train split x
    training_output_labels = tr_y[:, :48000]                                                # train split y

    benchmark_utility.benchmark_stop()                                                      # timer stop
    return Dataset(training_input_data, training_output_labels, validation_input_data, validation_output_labels, te_x_transposed, te_y)


# QUESTION 1_1_1 > SINGLE HIDDEN LAYER WITH 200 ReLU
async def train_architecture_Question1_1_1(dataset_obj, epochs = 50, learning_rate = 0.001):
    print("\n>>>\tSTARTING train_architecture_Question1_1_1 : 200 ReLU\t<<<")
    benchmark_utility.benchmark_start("train_architecture_Question1_1_1")
    
    # https://www.geeksforgeeks.org/introduction-deep-learning/
    weight_matrix_layer_1 = tf.Variable(tf.random.normal([200, 784], stddev = np.sqrt(2 / 784), dtype=tf.float32))          # init (200x784)
    bias_vector_layer_1 = tf.Variable(tf.zeros([200, 1], dtype=tf.float32))                                                 # init 1 (200x1)
    weight_matrix_layer_2 = tf.Variable(tf.random.normal([10, 200], stddev = 0.01, dtype=tf.float32))                       # init 2 (10x200)
    bias_vector_layer_2 = tf.Variable(tf.zeros([10, 1], dtype=tf.float32))                                                 # init 2 (10x1)
    
    training_loss_history = []                                                                                              # loss log
    # https://www.geeksforgeeks.org/deep-learning/adam-optimizer/
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)                                                  # init adam

    epoch_idx = 0
    while epoch_idx < epochs:
        # https://www.geeksforgeeks.org/backpropagation-in-neural-network/
        with tf.GradientTape() as tape_monitor:
            # https://www.geeksforgeeks.org/activation-functions-neural-networks/
            activation_hidden_layer_1 = tf.nn.relu(forward_pass(dataset_obj.training_input_data, weight_matrix_layer_1, bias_vector_layer_1))
            # https://www.geeksforgeeks.org/deep-learning/categorical-cross-entropy-in-multi-class-classification/
            predicted_probabilities = tf.nn.softmax(forward_pass(activation_hidden_layer_1, weight_matrix_layer_2, bias_vector_layer_2), axis = 0)
            current_loss_value = cross_entropy(predicted_probabilities, dataset_obj.training_output_labels)
        
        trainable_parameters = [weight_matrix_layer_1, bias_vector_layer_1, weight_matrix_layer_2, bias_vector_layer_2]
        
        gradients_computed = tape_monitor.gradient(current_loss_value, trainable_parameters)    # backprop gradients
        
        adam_optimizer.apply_gradients(zip(gradients_computed, trainable_parameters))           # update weights

        if epoch_idx % 10 == 0:
            val_hidden = tf.nn.relu(forward_pass(dataset_obj.validation_input_data, weight_matrix_layer_1, bias_vector_layer_1))
            val_probs = tf.nn.softmax(forward_pass(val_hidden, weight_matrix_layer_2, bias_vector_layer_2), axis = 0)
            val_acc = calculate_accuracy(val_probs, dataset_obj.validation_output_labels)
            
            training_loss_history.append(current_loss_value.numpy())
            
            print(f"\n>>>\tEPOCH {epoch_idx}, LOSS : {current_loss_value.numpy():.4f}, VAL ACC : {val_acc.numpy():.4f}\t<<<")
        
        epoch_idx += 1

    benchmark_utility.benchmark_stop()                                                              # timer stop

    return trainable_parameters, training_loss_history


# QUESTION 1_2_1 > TRIPLE LAYER ARCHITECTURE (128-128-10)
async def train_architecture_Question1_2_1(dataset_obj, epochs = 50, learning_rate = 0.001):
    print("\n>>>\tSTARTING Q1_2_1: Triple Layer (128-128-10)\t<<<")
    benchmark_utility.benchmark_start("train_architecture_Question1_2_1")
    
    # https://www.geeksforgeeks.org/deep-learning/neural-networks-a-beginners-guide/
    weight_matrix_layer_1 = tf.Variable(tf.random.normal([128, 784], stddev = np.sqrt(2 / 784), dtype=tf.float32))      # init 1
    bias_vector_layer_1 = tf.Variable(tf.zeros([128, 1], dtype=tf.float32))                                             # bias 1
    weight_matrix_layer_2 = tf.Variable(tf.random.normal([128, 128], stddev = np.sqrt(2 / 128), dtype=tf.float32))      # init 2
    bias_vector_layer_2 = tf.Variable(tf.zeros([128, 1], dtype=tf.float32))                                             # bias 2
    weight_matrix_layer_3 = tf.Variable(tf.random.normal([10, 128], stddev = 0.01, dtype=tf.float32))                   # init 3
    bias_vector_layer_3 = tf.Variable(tf.zeros([10, 1], dtype=tf.float32))                                              # bias 3

    training_loss_history = []
    # https://www.geeksforgeeks.org/deep-learning/adam-optimizer/
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    epoch_idx = 0
    while epoch_idx < epochs:
        with tf.GradientTape() as tape_monitor:
            activation_hidden_1 = tf.nn.relu(forward_pass(dataset_obj.training_input_data, weight_matrix_layer_1, bias_vector_layer_1))
            
            # MANUAL DROPOUT MASK
            dropout_rate = 0.2
            mask1 = tf.cast(tf.random.uniform(tf.shape(activation_hidden_1)) > dropout_rate, tf.float32)
            activation_hidden_1 = (activation_hidden_1 * mask1) / (1.0 - dropout_rate)

            activation_hidden_2 = tf.nn.relu(forward_pass(activation_hidden_1, weight_matrix_layer_2, bias_vector_layer_2))
            
            # MANUAL DROPOUT MASK
            mask2 = tf.cast(tf.random.uniform(tf.shape(activation_hidden_2)) > dropout_rate, tf.float32)
            activation_hidden_2 = (activation_hidden_2 * mask2) / (1.0 - dropout_rate)

            predicted_probabilities = tf.nn.softmax(forward_pass(activation_hidden_2, weight_matrix_layer_3, bias_vector_layer_3), axis = 0)
            current_loss_value = cross_entropy(predicted_probabilities, dataset_obj.training_output_labels)

        trainable_params = [weight_matrix_layer_1, bias_vector_layer_1, weight_matrix_layer_2, bias_vector_layer_2, weight_matrix_layer_3, bias_vector_layer_3]
        
        grads = tape_monitor.gradient(current_loss_value, trainable_params)
        
        adam_optimizer.apply_gradients(zip(grads, trainable_params))

        training_loss_history.append(current_loss_value.numpy())

        if epoch_idx % 10 == 0:
            val_act1 = tf.nn.relu(forward_pass(dataset_obj.validation_input_data, weight_matrix_layer_1, bias_vector_layer_1))
            val_act2 = tf.nn.relu(forward_pass(val_act1, weight_matrix_layer_2, bias_vector_layer_2))
            val_probs = tf.nn.softmax(forward_pass(val_act2, weight_matrix_layer_3, bias_vector_layer_3), axis = 0)
            val_acc = calculate_accuracy(val_probs, dataset_obj.validation_output_labels)

            print(f"\n>>>\tEPOCH {epoch_idx}, LOSS : {current_loss_value.numpy():.4f}, VAL ACC : {val_acc.numpy():.4f}\t<<<")
        
        epoch_idx += 1

    benchmark_utility.benchmark_stop()                                                      # timer stop
    
    return trainable_params, training_loss_history


# QUESTION 1_3_1 > MINI-BATCH OPTIMIZATION WITH BATCH SIZE 2048
async def train_architecture_Question1_3_1(dataset_obj, epochs = 10, learning_rate = 0.001, batch_size = 2048):
    print(f"\n>>> STARTING Question1_3_1: Mini-Batch {batch_size} <<<")
    benchmark_utility.benchmark_start("train_architecture_Question1_3_1")
    
    # https://www.geeksforgeeks.org/deep-learning/deep-learning-tutorial/
    weight_matrix_layer_1 = tf.Variable(tf.random.normal([200, 784], stddev = np.sqrt(2 / 784), dtype=tf.float32))              # init 1 (200x784)
    bias_vector_layer_1 = tf.Variable(tf.zeros([200, 1], dtype=tf.float32))                                                     # bias 1 (200x1)
    weight_matrix_layer_2 = tf.Variable(tf.random.normal([10, 200], stddev = 0.01, dtype=tf.float32))                           # init 2 (10x200)
    bias_vector_layer_2 = tf.Variable(tf.zeros([10, 1], dtype=tf.float32))                                                     # bias 2 (10x1)
    
    training_loss_history = []                                                                                                  # loss log
    # https://www.geeksforgeeks.org/deep-learning/adam-optimizer/
    adam_optimizer = tf.keras.optimizers.Adam(learning_rate = learning_rate)                                                   # init Adam
    total_samples_m = dataset_obj.training_input_data.shape[1]                                                                  # get M samples

    epoch_idx = 0
    while epoch_idx < epochs:
        # https://www.geeksforgeeks.org/data-preprocessing-machine-learning-python/
        random_shuffled_indices = tf.random.shuffle(tf.range(total_samples_m))                                                  # random indices
        input_shuffled = tf.gather(dataset_obj.training_input_data, random_shuffled_indices, axis = 1)                          # random x
        labels_shuffled = tf.gather(dataset_obj.training_output_labels, random_shuffled_indices, axis = 1)                      # random y

        i = 0
        while i < total_samples_m:                                                                                              # iterate batches
            input_batch = tf.cast(input_shuffled[:, i:i + batch_size], tf.float32)                                              # slice X batch
            label_batch = tf.cast(labels_shuffled[:, i:i + batch_size], tf.float32)                                              # slice Y batch
            
            with tf.GradientTape() as tape_monitor:
                # https://www.geeksforgeeks.org/activation-functions-neural-networks/
                activation_hidden_1 = tf.nn.relu(forward_pass(input_batch, weight_matrix_layer_1, bias_vector_layer_1))
                # https://www.geeksforgeeks.org/deep-learning/categorical-cross-entropy-in-multi-class-classification/
                predicted_probabilities = tf.nn.softmax(forward_pass(activation_hidden_1, weight_matrix_layer_2, bias_vector_layer_2), axis = 0)
                current_batch_loss = cross_entropy(predicted_probabilities, label_batch)
            
            trainable_params = [weight_matrix_layer_1, bias_vector_layer_1, weight_matrix_layer_2, bias_vector_layer_2]
            gradients_batch = tape_monitor.gradient(current_batch_loss, trainable_params)                                   # compute gradients
            adam_optimizer.apply_gradients(zip(gradients_batch, trainable_params))                                          # update step
            
            i += batch_size
        
        training_loss_history.append(current_batch_loss.numpy())                                                                # log batch loss
        print(f"\t>>> EPOCH {epoch_idx}\t>>> BATCH LOSS {current_batch_loss.numpy():.4f}")
        
        epoch_idx += 1

    benchmark_utility.benchmark_stop()                                                                                              # timer stop
    return trainable_params, training_loss_history


# PIPELINE
async def run_pipeline():
    print("\t>>> INITIALIZING DATA PIPELINE FOR GPU...\t")
    dataset_collection = await data_pipeline()                                                              # prep data
    
    with tf.device('/GPU:0'):                                                               # force DirectML GPU
        model_params_q1, loss_log_q1 = await train_architecture_Question1_1_1(dataset_collection)
        model_params_q2, loss_log_q2 = await train_architecture_Question1_2_1(dataset_collection)
        model_params_q3, loss_log_q3 = await train_architecture_Question1_3_1(dataset_collection)

    # TESTING EVALUATION
    testing_in = tf.cast(dataset_collection.testing_input_data, tf.float32)
    hidden_test = tf.nn.relu(forward_pass(testing_in, model_params_q2[0], model_params_q2[1]))
    hidden_test2 = tf.nn.relu(forward_pass(hidden_test, model_params_q2[2], model_params_q2[3]))
    final_probabilities = tf.nn.softmax(forward_pass(hidden_test2, model_params_q2[4], model_params_q2[5]), axis = 0)
    
    testing_accuracy = calculate_accuracy(final_probabilities, dataset_collection.testing_output_labels)
    print(f"\nTESTING ACCURACY Question1_2_1 Triple Layer >>> {testing_accuracy.numpy():.4f}")

    x_q1 = np.linspace(0, 50, len(loss_log_q1))
    x_q2 = np.linspace(0, 50, len(loss_log_q2))
    x_q3 = np.linspace(0, 50, len(loss_log_q3))

    plt.figure(figsize = (12, 6))                                                           # init plot
    plt.plot(x_q1, loss_log_q1, label = "Question1_1_1: Single Layer", linewidth = 2)
    plt.plot(x_q2, loss_log_q2, label = "Question1_2_1: Triple Layer", linewidth = 2)
    plt.plot(x_q3, loss_log_q3, label = "Question1_3_1: Mini-Batch 2048", linestyle = '-')
    plt.title("CPU VS GPU BENCHMARK")                                                       # plot title
    plt.xlabel("Checkpoint Interval")                                                       # axis x
    plt.ylabel("Cross Entropy Loss")                                                        # axis y
    plt.legend()                                                                            # plot legend
    plt.grid(True, alpha = 0.3)                                                             # add grid
    plt.show()                                                                              # render

if __name__ == "__main__":
    task.run(run_pipeline())                                                                # run pipeline