# Customer Churn Prediction using Artificial Neural Networks (ANN)

## 📋 Project Overview

This is a **production-ready Machine Learning system** that predicts customer churn using a deep learning Artificial Neural Network (ANN). The system is designed to help financial institutions identify high-risk customers who are likely to leave, enabling proactive retention strategies.

### Business & ML Perspective

**Business Goal:** Reduce customer attrition by identifying at-risk customers early, allowing businesses to implement targeted retention campaigns and improve customer lifetime value.

**ML Objective:** Build a binary classification model that predicts whether a customer will churn (exit) based on their demographic, behavioral, and financial attributes with high accuracy and interpretability.

---

## 🎯 Problem Statement

Customer churn is a critical metric in the banking and financial services industry. Understanding which customers are likely to leave helps businesses:

- Allocate resources for retention efforts efficiently
- Personalize customer engagement strategies
- Reduce revenue loss from departing customers
- Improve overall customer lifetime value

This project develops a supervised machine learning model that predicts churn probability on a customer-by-customer basis using their profile and account information.

---

## 📊 Dataset Description

### Dataset: `Churn_Modelling.csv`

**Size:** 10,000 customer records (10,001 rows including header)

**Features (13 input features):**

| Feature           | Type        | Description                                  |
| ----------------- | ----------- | -------------------------------------------- |
| `CreditScore`     | Numerical   | Customer's credit score (300-850)            |
| `Geography`       | Categorical | Customer's location (France, Germany, Spain) |
| `Gender`          | Categorical | Customer's gender (Male, Female)             |
| `Age`             | Numerical   | Customer's age in years (18-100)             |
| `Tenure`          | Numerical   | Years as a customer (0-10)                   |
| `Balance`         | Numerical   | Account balance in currency units            |
| `NumOfProducts`   | Numerical   | Number of products held (1-4)                |
| `HasCrCard`       | Binary      | Whether customer has a credit card (0/1)     |
| `IsActiveMember`  | Binary      | Whether customer is active (0/1)             |
| `EstimatedSalary` | Numerical   | Estimated annual salary                      |
| `RowNumber`       | ID          | Row identifier (dropped)                     |
| `CustomerId`      | ID          | Customer unique ID (dropped)                 |
| `Surname`         | ID          | Customer name (dropped)                      |

**Target Variable:**

- `Exited` (Binary): 1 = Customer churned, 0 = Customer retained

**Data Split:**

- Training set: 80% (8,000 samples)
- Test set: 20% (2,000 samples)

---

## 📁 Project Structure

```
annclassification/
│
├── README.md                          # Project documentation (this file)
├── requirements.txt                   # Python dependencies
│
├── Churn_Modelling.csv               # Raw dataset (10,000 customer records)
│
├── experiments.ipynb                  # Main notebook for model development
│                                     # - Data loading & exploration
│                                     # - Preprocessing & feature engineering
│                                     # - Model architecture definition
│                                     # - Training with callbacks
│
├── prediction.ipynb                   # Inference/prediction notebook
│                                     # - Model loading & testing
│                                     # - Single & batch predictions
│                                     # - Scenario-based analysis
│
├── app.py                            # Streamlit web application
│                                     # - Interactive UI for predictions
│                                     # - Real-time customer churn scoring
│
├── ann_model.h5                       # Trained TensorFlow/Keras model
│                                     # - Serialized neural network weights
│                                     # - Ready for inference
│
├── scaler.pkl                         # StandardScaler fitted on training data
│                                     # - For feature normalization
│
├── label_encoder_gender.pkl           # LabelEncoder for Gender feature
│                                     # - Maps: Female→0, Male→1
│
├── onehot_encoder_geography.pkl       # OneHotEncoder for Geography feature
│                                     # - Creates binary columns for regions
│
└── logs/                              # TensorBoard event logs
    └── fit/                           # Training history and metrics
        ├── 20260107-065104/          # Timestamped training session
        │   ├── train/
        │   └── validation/
        └── train/                     # Previous training sessions
```

---

## 🧠 Model Architecture

### ANN Overview

Artificial Neural Networks (ANNs) are inspired by biological neural systems and consist of interconnected layers of neurons that learn complex patterns from data through iterative weight adjustments.

### Network Architecture

The implemented model is a **Sequential Neural Network** with:

```
Input Layer (11 features after preprocessing)
    ↓
Dense Layer 1: 64 neurons, ReLU activation
    ↓
Dense Layer 2: 32 neurons, ReLU activation
    ↓
Output Layer: 1 neuron, Sigmoid activation
    ↓
Binary Classification (Churn or No Churn)
```

### Architecture Details

| Layer    | Units | Activation | Input Shape | Purpose                   |
| -------- | ----- | ---------- | ----------- | ------------------------- |
| Input    | -     | -          | 11 features | Raw preprocessed features |
| Hidden 1 | 64    | ReLU       | (11,)       | Learn complex patterns    |
| Hidden 2 | 32    | ReLU       | (64,)       | Further abstraction       |
| Output   | 1     | Sigmoid    | (32,)       | Binary probability (0-1)  |

### Activation Functions

- **ReLU (Rectified Linear Unit):** Used in hidden layers to introduce non-linearity and enable the network to learn complex relationships. Formula: $f(x) = \max(0, x)$

- **Sigmoid:** Used in output layer to convert raw predictions to probabilities between 0 and 1. Formula: $\sigma(x) = \frac{1}{1 + e^{-x}}$

### Loss & Optimization

- **Loss Function:** Binary Crossentropy - standard for binary classification problems

  - Formula: $L = -\frac{1}{n} \sum_{i=1}^{n} [y_i \log(\hat{y}_i) + (1-y_i) \log(1-\hat{y}_i)]$

- **Optimizer:** Adam (Adaptive Moment Estimation)
  - Combines advantages of AdaGrad and RMSProp
  - Adaptive learning rates for each parameter
  - Robust and efficient convergence

---

## 🔧 Data Preprocessing & Feature Engineering

### Step 1: Data Cleaning

- **Dropped irrelevant columns:** `RowNumber`, `CustomerId`, `Surname` (identifiers only, no predictive value)

### Step 2: Categorical Encoding

#### Label Encoding (Gender)

- **Purpose:** Convert categorical gender to numerical
- **Method:** Scikit-learn's `LabelEncoder`
- **Mapping:** Female → 0, Male → 1
- **Saved:** `label_encoder_gender.pkl`

#### One-Hot Encoding (Geography)

- **Purpose:** Convert geography categories to binary features
- **Method:** Scikit-learn's `OneHotEncoder`
- **Result:** 3 binary columns (Geography_France, Geography_Germany, Geography_Spain)
- **Saved:** `onehot_encoder_geography.pkl`

### Step 3: Feature Scaling

- **Method:** StandardScaler (Z-score normalization)
- **Formula:** $z = \frac{x - \mu}{\sigma}$ where $\mu$ = mean, $\sigma$ = standard deviation
- **Purpose:** Normalize all features to same scale (mean=0, std=1)
- **Benefit:** Improves neural network training stability and convergence
- **Saved:** `scaler.pkl`

### Step 4: Train-Test Split

- **Ratio:** 80% training, 20% testing
- **Random State:** 42 (for reproducibility)
- **Size:** 8,000 training, 2,000 test samples

### Final Feature Set

After preprocessing, the model receives **11 features**:

- CreditScore, Gender (encoded), Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography_France, Geography_Germany, Geography_Spain

---

## 📈 Training & Evaluation Strategy

### Training Configuration

| Parameter        | Value                      | Purpose                      |
| ---------------- | -------------------------- | ---------------------------- |
| Epochs           | 100                        | Maximum training iterations  |
| Batch Size       | 32                         | Samples per gradient update  |
| Validation Split | 20% of training data       | Monitor overfitting          |
| Early Stopping   | Patience=10                | Stop if val_loss plateaus    |
| Callbacks        | TensorBoard, EarlyStopping | Monitor and control training |

### Training Process

1. **Split data:** 80% train, 20% test
2. **Scale features:** Apply StandardScaler normalization
3. **Initialize model:** Sequential ANN with specified architecture
4. **Compile:** Adam optimizer + binary crossentropy loss
5. **Train:** With validation monitoring and early stopping
6. **Save:** Model weights to `ann_model.h5`

### Early Stopping Strategy

- **Monitor:** Validation loss
- **Patience:** 10 epochs without improvement
- **Action:** Restore best weights and stop training
- **Benefit:** Prevent overfitting and unnecessary computation

### Monitoring with TensorBoard

- Real-time visualization of training/validation metrics
- Loss curves, accuracy progression, computational graph
- Event logs stored in `logs/fit/` with timestamps

---

## 📊 Results & Metrics

### Model Performance Indicators

The trained model is evaluated on multiple metrics:

- **Accuracy:** Percentage of correct predictions (both classes)
- **Precision:** Of customers predicted as churners, how many actually churn
- **Recall:** Of actual churners, how many the model identifies
- **F1-Score:** Harmonic mean balancing precision and recall

### Key Insights from Testing

**Model Outputs:**

- **Churn Probability:** Continuous value 0-1 output by sigmoid activation
- **Decision Threshold:** 0.5 (customers with probability > 0.5 predicted as churners)

**Tested Scenarios:**

- Low income + inactive members → High churn risk
- Advanced age (60+) + low tenure → High churn risk
- Germany location + female gender → Elevated churn indicators
- High credit score + active membership → Low churn risk

### TensorBoard Logs

Training visualization available in `logs/fit/` directory with event files capturing:

- Training loss progression
- Validation loss progression
- Accuracy metrics over epochs
- Training date: 2026-01-07 06:51:04

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Conda (optional, for environment management)

### Step 1: Clone/Navigate to Project

```bash
cd /home/mohamed-tamer/Downloads/krish-naik/Starter/2.MachineLearning/annclassification
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n churn_prediction python=3.10
conda activate churn_prediction
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Libraries

```
tensorflow==2.20.0       # Deep learning framework
pandas                   # Data manipulation
numpy                    # Numerical computing
scikit-learn            # ML preprocessing & utilities
tensorboard             # Training visualization
matplotlib              # Data visualization
streamlit               # Web app framework
scikeras                # Scikit-learn wrapper for Keras
ipykernel               # Jupyter kernel
```

### Step 4: Verify Installation

```bash
python -c "import tensorflow as tf; print(f'TensorFlow version: {tf.__version__}')"
```

---

## 🎮 How to Run the Project

### Option 1: Interactive Web Application (Recommended)

```bash
streamlit run app.py
```

**Features:**

- User-friendly interface for real-time predictions
- Dropdown selections for categorical features
- Slider inputs for numerical ranges
- Instant churn probability calculation
- Visual feedback (success/error messages)

**Access:** Browser opens at `http://localhost:8501`

### Option 2: Jupyter Notebooks

#### Explore Training Process

```bash
jupyter notebook experiments.ipynb
```

- View data preprocessing steps
- See model architecture definition
- Monitor training history
- Analyze TensorBoard visualizations

#### Run Predictions

```bash
jupyter notebook prediction.ipynb
```

- Load trained model and encoders
- Make single-customer predictions
- Test multiple scenarios
- Analyze churn probability patterns

### Option 3: Python Script (Custom Inference)

```python
import tensorflow as tf
import pickle
import pandas as pd

# Load artifacts
model = tf.keras.models.load_model("ann_model.h5")
scaler = pickle.load(open("scaler.pkl", "rb"))
label_encoder = pickle.load(open("label_encoder_gender.pkl", "rb"))
ohe = pickle.load(open("onehot_encoder_geography.pkl", "rb"))

# Prepare input
input_dict = {
    "CreditScore": 750,
    "Geography": "France",
    "Gender": "Male",
    "Age": 35,
    "Tenure": 5,
    "Balance": 100000,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 75000
}

# Predict
# [preprocessing steps here...]
prediction = model.predict(scaled_input)
churn_probability = prediction[0][0]
```

---

## 🛠️ Technologies Used

### Core Libraries

| Technology           | Version | Purpose                                 |
| -------------------- | ------- | --------------------------------------- |
| **TensorFlow/Keras** | 2.20.0  | Deep learning framework, model training |
| **scikit-learn**     | Latest  | Data preprocessing, train-test split    |
| **pandas**           | Latest  | Data manipulation, feature engineering  |
| **NumPy**            | Latest  | Numerical computing, matrix operations  |
| **Streamlit**        | Latest  | Web application frontend                |
| **TensorBoard**      | Latest  | Training visualization & monitoring     |

### Key Techniques

- **Deep Learning:** Multi-layer perceptron (MLP)
- **Activation Functions:** ReLU, Sigmoid
- **Optimization:** Adam optimizer with adaptive learning rates
- **Regularization:** Early stopping (prevent overfitting)
- **Encoding:** Label encoding, One-Hot encoding
- **Scaling:** StandardScaler (Z-score normalization)

---

## 🔍 Model Assumptions & Limitations

### Assumptions

1. **Data Independence:** Each customer record is independent (no temporal dependencies)
2. **Feature Relevance:** Selected features are sufficient to predict churn
3. **Stationary Distribution:** Training and deployment data follow similar distributions
4. **Binary Classification:** Churn is binary (churned or retained, no partial churn)
5. **Balanced Threshold:** 0.5 probability threshold suitable for the business context

### Limitations

1. **No Temporal Information:** Model doesn't capture trends over time or seasonality
2. **Feature Engineering:** Limited interaction features (current features are mostly independent)
3. **Imbalanced Data:** Not addressed in current implementation (may affect minority class predictions)
4. **Cold Start Problem:** Cannot predict churn for very new customers with minimal tenure
5. **Black Box Nature:** Neural networks provide limited interpretability (difficult to explain individual predictions)
6. **Data Drift:** Model assumes deployment environment matches training data distribution
7. **Fixed Feature Set:** Requires exact same features during inference; changes require retraining

---

## 💡 Strengths

✅ **Strong Empirical Performance:** Neural networks capture complex non-linear relationships effectively

✅ **Production Ready:** Complete pipeline with preprocessing, training, and deployment code

✅ **Well Organized:** Clear separation of concerns (notebooks, app, artifacts)

✅ **Interpretable Pipeline:** Preprocessing steps clearly documented and reproducible

✅ **Monitoring Capability:** TensorBoard integration for real-time training visualization

✅ **Easy Deployment:** Streamlit app provides simple user interface for predictions

✅ **Reproducible:** Fixed random seed and saved artifacts ensure consistent results

✅ **Multiple Interaction Modes:** Notebooks for exploration, app for production, scripts for automation

---

## 🚀 Future Improvements

### Model Architecture

1. **Regularization Techniques**

   - Add dropout layers to reduce overfitting
   - Implement L1/L2 regularization on weights
   - Experiment with batch normalization

2. **Hyperparameter Tuning**

   - Grid search or Bayesian optimization for optimal architecture
   - Vary layer sizes, learning rates, batch sizes
   - Tune early stopping patience parameter

3. **Ensemble Methods**
   - Combine multiple ANN models
   - Use gradient boosting (XGBoost, LightGBM) alongside ANN
   - Implement voting/stacking mechanisms

### Feature Engineering

4. **Advanced Feature Creation**

   - Polynomial features (age², salary/balance ratio)
   - Interaction terms (age × tenure, balance × products)
   - Behavioral segments based on product usage

5. **Temporal Features**

   - If historical data available: transaction frequency, velocity
   - Seasonal patterns in account activity
   - Trend indicators

6. **Class Imbalance Handling**
   - SMOTE (Synthetic Minority Oversampling)
   - Class weights in loss function
   - Threshold adjustment based on business costs

### Evaluation & Deployment

7. **Advanced Validation**

   - Cross-validation (k-fold) for robust performance estimates
   - ROC-AUC curves and PR curves for threshold optimization
   - Confusion matrix analysis and F1-score optimization

8. **Model Interpretability**

   - SHAP values for feature importance
   - LIME for local explanations
   - Attention mechanisms for transparency

9. **Production Pipeline**

   - API endpoints (Flask/FastAPI) for microservices
   - Database integration for logging predictions
   - Model versioning and A/B testing framework
   - Automated retraining on new data

10. **Monitoring & Maintenance**
    - Data drift detection
    - Model performance degradation alerts
    - Feedback loops for continuous improvement
    - Model explainability dashboard

---

## 📝 Usage Examples

### Example 1: High-Risk Customer (Likely to Churn)

```
Input:
- Geography: Germany
- Gender: Female
- Age: 65 years
- Tenure: 2 years
- Balance: $0
- IsActiveMember: No

Output: Churn Probability = 85% → Customer is at HIGH RISK
Recommendation: Initiate retention campaign immediately
```

### Example 2: Low-Risk Customer (Likely to Retain)

```
Input:
- Geography: France
- Gender: Male
- Age: 35 years
- Tenure: 8 years
- Balance: $150,000
- IsActiveMember: Yes

Output: Churn Probability = 15% → Customer is at LOW RISK
Recommendation: Standard engagement strategy
```

---

## 📚 References & Resources

### Deep Learning Fundamentals

- [TensorFlow Official Documentation](https://www.tensorflow.org/)
- [Keras API Reference](https://keras.io/)
- [Understanding Neural Networks](https://colah.github.io/posts/2014-03-NN-Manifold-Topology/)

### Churn Prediction Research

- Classic approaches to customer churn prediction in banking
- Impact of feature engineering on model performance
- Business metrics for evaluating retention campaigns

### Tools & Frameworks

- [Streamlit Documentation](https://docs.streamlit.io/)
- [TensorBoard Guide](https://www.tensorflow.org/tensorboard)
- [Scikit-learn Preprocessing](https://scikit-learn.org/stable/modules/preprocessing.html)

---

**Key Components:**

- Training notebook: `experiments.ipynb` - Data preprocessing & model development
- Inference notebook: `prediction.ipynb` - Testing & scenario analysis
- Deployment: `app.py` - Streamlit web interface

### Learning Objectives Addressed

✓ Understanding ANN architectures and hyperparameters
✓ Data preprocessing and feature engineering best practices
✓ Model training with callbacks and monitoring
✓ Deploying ML models to production (web app)
✓ Making predictions on new data with proper scaling

---

## 📞 Questions & Support

For questions about:

- **Model Architecture:** Review the "Model Architecture" section above
- **Feature Processing:** See "Data Preprocessing & Feature Engineering"
- **Running Predictions:** Check "How to Run the Project"
- **Improving Performance:** Explore "Future Improvements" section

---

## 📄 License

This project is part of an educational course. Use for learning and non-commercial purposes.

---

**Last Updated:** January 7, 2026  
**Status:** ✅ Production Ready
