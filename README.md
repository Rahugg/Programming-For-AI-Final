# Car Price Prediction 

## Project Overview

This project builds a machine learning model that predicts the price of a car based on its characteristics (year, mileage, power, dimensions, fuel type, transmission, and more).

We implemented a full ML pipeline:

- dataset loading and preprocessing  
- feature engineering and feature selection  
- training multiple regression models  
- comparing performance  
- selecting and saving the best model  
- generating visualizations  

The project uses real-world car data (2,059 rows, 20+ features).

The best model is **XGBoost**, reaching **R² ≈ 0.907** — meaning the model explains ~90% of price variance.

---

## Project Goals

- Build a system that predicts car prices automatically  
- Compare ML algorithms and choose the best  
- Perform feature selection to understand what affects price  
- Visualize model performance  
- Create a structured, reproducible Python project  

---

## Technologies Used

### **Programming & Data**
- Python 3.x  
- Pandas — working with tabular data  
- NumPy — numerical operations  

### **Machine Learning**
- Scikit-learn (preprocessing, metrics, models)  
- XGBoost — boosting model with highest performance  
- RandomForest — strong baseline tree model  
- SVR — regression via support vectors  
- HuberRegressor — robust linear model  
- DummyRegressor — baseline for comparison  

### **Feature Selection**
- SelectKBest  
- RFE (Recursive Feature Elimination)  
- PCA (Principal Component Analysis)

### **Visualization**
- Matplotlib  
- Custom plots: Actual vs Predicted, Feature Importance  

### **Model Saving**
- joblib (saving best model to disk)

---

##  ML Pipeline

The project follows a structured pipeline:

1. **Dataset auto-detection**  
2. **Cleaning & preprocessing**  
3. **Train-test split**  
4. **Feature encoding & scaling**  
5. **Feature selection (KBest / RFE / PCA)**  
6. **Model training**  
7. **Model comparison (MAE, RMSE, R²)**  
8. **Selecting the best model (XGBoost)**  
9. **Saving model → `models/best_model.joblib`**  
10. **Generating plots (optional)**

---

## Project Structure

``` 
project/
│── data/ # dataset (cardetailsv4.csv)
│── models/ # saved trained model(s)
│── reports/ # generated visualizations
│── utils/ # preprocessing & helper functions
│── ui/ # optional graph utilities
│── main.py # runs the full ML pipeline
│── make_plots.py # optional script for generating plots
│── requirements.txt # dependencies
└── README.md # project documentation
```

###  Key Files

- **main.py** — runs the entire ML pipeline automatically  
- **make_plots.py** — *optional* helper for generating visualizations  
- **models/best_model.joblib** — best trained model  
- **reports/** — contains actual_vs_predicted.png and feature_importance.png  

---

## How to Run the Project

### Install dependencies

pip install -r requirements.txt

### Run full pipeline

python main.py

main.py will automatically:

- load dataset

- preprocess data

- perform feature selection

- train multiple models

- evaluate them

- save best model to models/best_model.joblib

- print metrics in the console

### Generate graphs
python make_plots.py


### Output:

``` reports/
 ├── actual_vs_predicted.png
 └── feature_importance.png
```


