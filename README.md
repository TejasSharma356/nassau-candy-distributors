# 🍬 Nassau Candy Distributor — Shipping Route Efficiency Dashboard

> **Unified Mentor | Data Analytics & Machine Learning Project**  
> An interactive business intelligence platform for analyzing and predicting shipping logistics performance.

---

## 📌 Project Overview

Nassau Candy Distributor is one of the largest confectionery wholesale distributors in the United States. This project delivers a full-stack analytics and ML platform built to help their operations team:

- Monitor shipping route efficiency across the entire distribution network
- Identify geographic bottlenecks and congestion-prone states
- Compare cost-time tradeoffs across shipping methods
- **Predict whether an order will be delayed** using a trained ML model

The dashboard is built with **Streamlit** and **Plotly**, backed by a **Random Forest** machine learning model trained on historical order data.

---

## ✨ Features

| Tab | Feature |
|---|---|
| 📊 Route Efficiency | Top 10 & Bottom 10 routes by efficiency score |
| 🗺️ Geographic Map | US choropleth map + Geographic Bottleneck Scatter Plot |
| 📦 Ship Mode Analysis | Lead time distributions, delay rates, cost-time tradeoffs |
| 🔬 Route Drill-Down | Deep-dive into a specific Factory → State route |
| 🤖 Predict Delays | Live ML-powered order delay predictor with probability gauge |

### 🤖 Machine Learning — Delay Predictor
- **Model**: Random Forest Classifier (`scikit-learn`)
- **Accuracy**: ~83% on the held-out test set
- **Recall for Delays**: 83% (catches 8 out of 10 actual delays)
- **Features Used**: Region, State, Factory, Ship Mode, Sales, Units, Order Month, Order Day of Week, Order Quarter
- **Output**: Delay probability (0–100%) + Top 5 key delay drivers chart

---

## 🗂️ Project Structure

```
nassau-candy-distributors/
│
├── app/
│   └── app.py                  # Main Streamlit dashboard application
│
├── src/
│   ├── data_processing.py      # Data ingestion, cleaning & lead time calculation
│   ├── analytics.py            # Route KPI computation & efficiency scoring
│   └── ml_model.py             # ML model training, feature engineering & inference
│
├── data/
│   ├── raw/
│   │   ├── Nassau Candy Distributor.csv   # Original orders dataset
│   │   ├── factory_coordinates.csv        # Factory lat/lon for map markers
│   │   └── product_factories.csv          # Product → Factory mapping
│   └── processed/
│       ├── cleaned_orders.csv             # Cleaned & feature-engineered orders
│       └── route_kpis.csv                 # Pre-computed route KPI scores
│
├── notebooks/                  # Exploratory analysis notebooks
├── requirements.txt            # Python dependencies
└── README.md
```

---

## ⚙️ Setup & Installation

### Prerequisites
- Python 3.9+
- pip

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/nassau-candy-distributors.git
cd nassau-candy-distributors
```

### 2. Create & Activate a Virtual Environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Data Pipeline
Run these scripts **once** to process the raw data and train the ML model:
```bash
python src/data_processing.py
python src/analytics.py
python src/ml_model.py
```

### 5. Launch the Dashboard
```bash
streamlit run app/app.py
```

The app will open automatically at `http://localhost:8501`.

---

## 📊 Dashboard Walkthrough

### Sidebar Filters
Use the left sidebar to interactively filter all charts by:
- **Date Range** (Order Date)
- **Region**
- **State / Province**
- **Ship Mode** (Standard, Second Class, First Class, Same Day)
- **Delay Threshold** (number of days to define a "delay")

### 🤖 Predict Delays Tab
1. Select the **Region**, **State**, **Factory**, and **Ship Mode**
2. Enter the estimated **Sales ($)** and **Units**
3. Click **Predict Delay Risk**
4. View the **Delay Probability Gauge** and **Top Feature Drivers** chart

---

## 🧠 ML Model Details

| Metric | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Training Split | 80% train / 20% test |
| Overall Accuracy | ~83% |
| Delay Recall | 83% |
| Delay Precision | 45% |
| Features | 9 engineered features |

> **Design Philosophy:** The model is tuned to prioritize **recall over precision** — it will alert you about *potential* delays liberally so you can take proactive action, even if some alerts are false alarms. In supply chain logistics, missing a genuine delay is far more costly than an unnecessary warning.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Streamlit** | Interactive web dashboard |
| **Plotly** | All interactive charts & maps |
| **Pandas / NumPy** | Data processing & feature engineering |
| **scikit-learn** | Random Forest ML model |
| **joblib** | Model serialization |
| **matplotlib** | DataFrame styling dependency |

---

## 📄 License

This project was developed as part of the **Unified Mentor Internship Program**.

---

## 👤 Author

**Tejas**  
Unified Mentor Data Analytics Intern
