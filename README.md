# 🧱 Concrete Compressive Strength Prediction

A machine learning project that predicts concrete compressive strength (MPa) from mix design inputs, with a deployed Streamlit web app.

## 🔗 Live App
[Try the live predictor here](YAHAN_APNA_STREAMLIT_URL_PASTE_KARO)

## 📌 Problem Statement
Concrete compressive strength is a highly nonlinear function of its ingredients and curing age. Traditional trial-batch testing to determine mix strength is time-consuming and costly. This project builds a regression model to predict strength directly from mix design values, helping engineers estimate outcomes before physical testing.

## 📊 Dataset
- **Source:** [UCI ML Repository — Concrete Compressive Strength Dataset](https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength)
- **Size:** 1,030 samples (1,005 after removing duplicates), 8 input features + 1 target
- **Features:** Cement, Blast Furnace Slag, Fly Ash, Water, Superplasticizer, Coarse Aggregate, Fine Aggregate, Age (days)
- **Target:** Concrete compressive strength (MPa)

## 🛠️ Approach
1. Data cleaning (removed 25 duplicate rows)
2. Exploratory Data Analysis — distributions, correlation heatmap, scatter plots
3. Train-test split (80/20) with feature scaling (StandardScaler)
4. Trained and compared 3 models: Linear Regression, Random Forest, XGBoost
5. Selected best model based on R², RMSE, MAE
6. Feature importance analysis
7. Deployed as an interactive Streamlit app

## 📈 Results

| Model              | R²     | RMSE    | MAE    |
|--------------------|--------|---------|--------|
| Linear Regression   | 0.5802 | 11.1913 | 8.8953 |
| Random Forest        | 0.9076 | 5.2496  | 3.4835 |
| **XGBoost (Best)**  | **0.9243** | **4.7517** | **2.9306** |

**Best Model: XGBoost** — explains 92.4% of the variance in compressive strength.

## 🔑 Key Insight — Feature Importance
**Age** and **Cement** are the strongest predictors of strength, consistent with concrete engineering fundamentals: strength develops through cement hydration over time.

## 💻 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/satyapalpatel51-del/concrete-strength-prediction.git
cd concrete-strength-prediction

# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
# source venv/bin/activate  # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Train the model
python -m src.train_model

# Run the app
streamlit run app.py
```

## 🗂️ Project Structure

concrete-strength-prediction/
├── data/
│   └── concrete_data.csv
├── notebooks/
│   └── notebook.ipynb
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── evaluate.py
├── app.py
├── requirements.txt
└── README.md
