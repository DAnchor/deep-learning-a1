# Hardware Benchmarking of Fashion MNIST on AMD GPU (DirectML)

This repository provides a low-level implementation of Deep Neural Networks utilizing the **TensorFlow GradientTape API**, specifically optimized for **AMD GPU hardware** via the **DirectML backend**. The system benchmarks various architectures and optimization strategies as **Mini-Batch** (Batch Size: 2048) approaches.

---

## 1. Environment Configuration (DirectML)

To interface with AMD hardware, the environment utilizes the DirectML plugin to leverage the DirectX 12 API for high-performance tensor computation.

### Technical Requirements
* **Python**: 3.10.x
* **TensorFlow**: 2.1x (with `tensorflow-directml-plugin`)
* **Device Allocation**: Operations are explicitly forced to `'/GPU:0'` (Line 277) to ensure utilization of the DirectML device.

---

## 2. Network Architectures

The models are built using raw tensor operations and manual weight initialization to demonstrate low-level logic, avoiding high-level Keras abstractions.

| Requirement | Code Location | Implementation Detail |
| :--- | :--- | :--- |
| **Question 1_1_1** | Line 117 | Single hidden layer architecture with 200 ReLU neurons. |
| **Question 1_2_1** | Line 164 | Triple layer architecture (128-128-10 configuration). |
| **Question 1_3_1** | Line 223 | Mini-Batch optimization utilizing a batch size of 2048. |
| **Activation Functions**| Lines 136 and boyond | `tf.nn.relu` for hidden layers; `tf.nn.softmax` for output. |
| **Vectorization** | Lines 59: | `tf.matmul` facilitates Z = WX + b across all implementations. |

---

## 3. Core Mandatory Functions

### A. Forward Pass
The `forward_pass` function (Lines 59:) manages forward propagation. It calculates the linear score function and is optimized for GPU memory access patterns by using transposed input matrices.

### B. Cross-Entropy Cost
Defined at (Lines 64:), the cost is calculated manually using `tf.math.log` and `tf.reduce_sum`. A `numerical_stability_epsilon` (1 * 10^-7) is integrated to prevent logarithmic errors during backpropagation.

### C. Accuracy Assessment
The `calculate_accuracy` function (Lines 71:) utilizes `tf.argmax` to extract class indices from the probability distribution, which are then compared against ground truth labels to return a percentage.

---

## 4. Manual Dropout Implementation

To mitigate overfitting in the deeper architecture (**Question 1_2_1**), a **Manual Dropout Mask** is applied via direct tensor manipulation rather than high-level layers.

* **Mask Generation**: A binary mask is created using `tf.random.uniform` (Lines 187:).
* **Inverted Dropout**: Active neurons are scaled by 1 / (1 - dropout_rate) to maintain consistent expected values.

---

## 5. Optimization and Training Logic

Training is handled through **Automatic Differentiation** within the `tf.GradientTape` context.

* **Gradient Recording**: Tape monitors operations at (Lines 139, 177, and 238).
* **Manual Backprop**: Derivatives are extracted via `tf.GradientTape` (Lines 143, 193, and 246).
* **Adam Optimizer**: Weights are updated using the Adam algorithm (Lines 129:, 178, 235) for adaptive learning rate management.

---

## 6. Mini-Batch Optimization (Question 1_3_1)

To fully saturate the AMD GPU's compute units, the pipeline implements a high-throughput mini-batch approach.

* **Shuffling**: Data indices are randomized every epoch using `tf.random.shuffle` and `tf.gather` (Lines 241:).
* **Partitioning**: Samples are processed in chunks of **2048** (Lines 223:), significantly reducing the overhead associated with the DirectML bridge compared to small batch sizes.

---

## 7. Benchmarking and Visualization

The final evaluation (Lines 273:) generates a comparative analysis of the different training configurations.

* **Convergence Analysis**: The Mini-Batch 2048 approach demonstrates a more stable and efficient reduction in cross-entropy loss.
* **Hardware Benchmarking**: The results illustrate the performance delta between standard single-layer execution and optimized multi-layer GPU-bound training.
