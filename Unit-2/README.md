# HCL Unit-2 Assignment: Multiclass Sentiment Analysis & Automated Hyperparameter Tuning

## 1. Project Objective
Perform 3-class sentiment analysis (`Negative (0)`, `Neutral (1)`, `Positive (2)`) on the Hugging Face **`Sp1786/multiclass-sentiment-analysis-dataset`** (`31,232` train, `5,205` validation, and `5,206` test samples) by building, automatically tuning, and comparing four architectures:
1. **Dense Neural Network (ANN)** (`Neural_Network.ipynb`)
2. **Simple Recurrent Neural Network (SimpleRNN)** (`SimpleRNN.ipynb`)
3. **Long Short-Term Memory (LSTM)** (`LSTM.ipynb`)
4. **Gated Recurrent Unit (GRU)** (`GRU.ipynb`)

---

## 2. Predefined Hyperparameter Search Space (`KerasTuner RandomSearch`)
Each architecture was automatically tuned across predefined hyperparameter ranges using explicit Backpropagation (`tf.GradientTape` with gradient clipping at `1.0`):
- **Embedding Dimension:** `[64, 128, 256]`
- **Hidden / Recurrent Units:** `[64, 128, 256, 512]`
- **Dropout Rate:** `0.2` to `0.5` (`step = 0.1`)
- **Learning Rate (Adam Optimizer):** `[0.1, 0.01, 0.001]`

---

## 3. Comparative Model Performance & Tuning Results

| Model Architecture | Notebook File | Optimal Units | Optimal Dropout | Optimal Learning Rate | Peak Val Accuracy (ROC-AUC %) | Exact-Match F1 Accuracy | True Predictions (Neg / Neu / Pos) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Neural Network (ANN)** | `Neural_Network.ipynb` | 256 | 0.3 / 0.4 | 0.001 | ~84.50% | ~67% | Baseline N-Gram TF-IDF |
| **2. SimpleRNN (Bi-RNN)** | `SimpleRNN.ipynb` | 64 / 128 | 0.3 | 0.001 | **85.51%** | **67%** | `915` / `1248` / `1334` |
| **3. LSTM (Bi-LSTM)** | `LSTM.ipynb` | **128** | **0.3** | **0.001** | **87.08% (Winner)** | **70% (Winner)** | `949` / `1398` / `1308` |
| **4. GRU (Conv1D + Bi-GRU)** | `GRU.ipynb` | 64 / 128 | 0.3 | 0.001 | **85.25%** | **68%** | **`978`** / `1319` / `1226` |

### Overall Architecture Performance Comparison
<img width="1289" height="440" alt="Comparing chart" src="https://github.com/user-attachments/assets/9522c496-8ce0-4f85-9075-f5f412d3d12d" />

---

## 4. Best-Performing Model & Key Conclusions
- **Best Overall Model:** **Bidirectional LSTM (`LSTM.ipynb`)** achieved the highest validation accuracy (**87.08%**) and highest multiclass F1-score (**0.70**), accurately classifying `3,655` out of `5,206` unseen test samples (`949` Negative, `1,398` Neutral, and `1,308` Positive).
- **Why LSTM Won:** Its three-gate memory architecture (**Forget, Input, and Output gates**) alongside dual pooling (`GlobalAveragePooling1D` + `GlobalMaxPooling1D`) preserved long-range context across 100-word sequences without suffering from the vanishing gradient problem seen in `SimpleRNN.ipynb`.
- **Best Hyperparameter Configuration:**
  * **Learning Rate:** `0.001` consistently outperformed `0.01` and `0.1` across all four architectures (high learning rates like `0.1` caused gradient instability).
  * **Recurrent Units:** `128` units with a **Dropout rate of `0.3`** provided the ideal balance between feature learning and regularization.
- **GRU Highlight (`GRU.ipynb`):** Combining contraction expansion, balanced class weights, and `Conv1D` with `GRU` achieved the highest **Negative class sensitivity (`978` true negatives)** while training ~25% faster per epoch than LSTM.

---

## 5. Visual Confusion Matrices & Hyperparameter Tuning Charts

### 1. Dense Neural Network (`Neural_Network.ipynb`)
<img width="1389" height="490" alt="Neural_Network" src="https://github.com/user-attachments/assets/39ac9756-9677-40d7-9492-596db22a54e6" />

### 2. SimpleRNN (`SimpleRNN.ipynb`)
<img width="1389" height="490" alt="RNN" src="https://github.com/user-attachments/assets/877c7f2a-df39-409b-84c9-4115687496ff" />

### 3. Bidirectional LSTM — Winner (`LSTM.ipynb`)
<img width="1389" height="490" alt="LSTM" src="https://github.com/user-attachments/assets/2b2972f3-bddd-4b81-bb88-c600c02adae9" />

### 4. Bidirectional GRU (`GRU.ipynb`)
<img width="1389" height="490" alt="GRU" src="https://github.com/user-attachments/assets/b818318b-8c42-45fc-98d0-75e122ab5eb4" />
