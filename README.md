# Weather Prediction Model

This project builds a **machine learning model** to predict whether it will **rain tomorrow** based on historical Australian weather data.
The dataset used is `weatherAUS.csv`, which contains daily weather observations from multiple locations across Australia.

---

## 🌦️ Project Overview

The goal of this project is to apply data preprocessing, exploratory analysis, and machine-learning classification techniques to predict the target variable:

* `RainTomorrow` → Yes/No

The workflow includes:

1. Data loading & cleaning
2. Handling missing values
3. Feature encoding
4. Train-test split
5. Model training (RandomForestClassifier)
6. Model evaluation
7. Predictions on unseen data

---

## 📁 Project Structure

```
weather-prediction-model/
│── main.py               # Main training + prediction script
│── weatherAUS.csv        # Dataset
│── requirements.txt      # Python dependencies
│── README.md             # Project documentation
```

---

## 🛠️ Technologies Used

* Python
* NumPy
* Pandas
* Matplotlib
* Seaborn
* Scikit-learn

---

## 🚀 How to Run the Project

### 1. Clone the repository

```bash
git clone https://github.com/Amar0703/weather-prediction-model.git
cd weather-prediction-model
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the model

```bash
python main.py
```

This will:

* Load and preprocess the dataset
* Train the Random Forest classifier
* Print model accuracy and metrics

---

## 📊 Dataset Information

The dataset includes weather-related attributes such as:

* Temperature
* Humidity
* Rainfall
* WindSpeed
* Pressure
* Evaporation
* Cloud cover
* RainToday
* RainTomorrow (Target)

---

## 🔍 Model Used

**RandomForestClassifier**

Chosen because:

* Handles categorical + numerical features
* Performs well on noisy real-world data
* Reduces overfitting using ensemble learning

---

## 📈 Results

The model outputs:

* Accuracy score
* Classification report
* Confusion matrix

(Actual values depend on execution.)

---
