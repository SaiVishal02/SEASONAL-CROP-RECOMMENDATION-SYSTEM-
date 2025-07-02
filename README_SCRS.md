
# 🌾 Seasonal Crop Recommendation System (SCRS)

The Seasonal Crop Recommendation System is a Streamlit-based machine learning web app that helps farmers choose the best crops based on their land’s soil and environmental conditions. It leverages a trained Random Forest model for accurate and data-driven crop predictions.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Modules Used](#modules-used)
- [How It Works](#how-it-works)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [Streamlit Interface](#streamlit-interface)
- [Testing](#testing)
- [Future Scope](#future-scope)
- [Contributors](#contributors)
- [License](#license)

---

## 🌟 Project Overview

This system helps farmers make informed decisions on crop selection based on soil nutrients (N, P, K), temperature, pH, humidity, rainfall, season, and soil type. It bridges traditional farming with modern machine learning, enhancing productivity, sustainability, and efficiency.

---

## ✅ Features

- Predicts suitable crops based on real input data.
- Interactive and simple Streamlit web UI.
- Supports major Indian soil types and seasons.
- Powered by Random Forest Classifier with high accuracy.
- Easily extendable and maintainable.

---

## 🧰 Tech Stack

- **Frontend**: Streamlit
- **Backend**: Python, scikit-learn
- **Machine Learning**: Decision Tree, Random Forest, Logistic Regression
- **Deployment**: Local / Streamlit Cloud

---

## 📦 Modules Used

- `pandas`: Data handling
- `numpy`: Numerical operations
- `scikit-learn`: ML modeling
- `streamlit`: Web app interface
- `pickle`: Model saving/loading
- `matplotlib`, `seaborn`: Visualization

---

## 🔁 How It Works

1. Input parameters via UI (e.g., NPK levels, temperature, season).
2. Preprocessed data is passed to a trained ML model.
3. The model predicts the most suitable crop.
4. Output is shown on screen in real-time.

---

## ⚙️ Installation

```bash
git clone https://github.com/yourusername/seasonal-crop-recommendation
cd seasonal-crop-recommendation
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

---

## 🖥️ Usage

- Go to your browser: `http://localhost:8501`
- Fill in sidebar inputs like soil type, season, NPK values, etc.
- Click **Recommend Crop**
- View the recommended crop on the screen

---

## 🧠 Model Training

- Dataset: [Kaggle Crop Recommendation Dataset](https://www.kaggle.com/datasets/varshitanalluri/crop-recommendation-dataset)
- Algorithms used: Decision Tree, Random Forest, Logistic Regression
- Evaluation: Accuracy, Cross-validation, Confusion Matrix
- Final Model: Random Forest (saved as `RandomForestClassifier.pkl`)

---

## 🖼️ Streamlit Interface

- Sidebar for user inputs (soil, NPK, temp, etc.)
- Center display for results
- Real-time prediction based on model

---

## 🧪 Testing

- Unit tests for preprocessing and model functions
- Integration tested between UI and backend model
- Accuracy tested with unseen data
- Feedback collected via user trials

---

## 🔮 Future Scope

- Add crop yield estimation
- Use geolocation for weather and soil data
- Expand to more regions/seasons
- Add multilingual UI
- Deploy on cloud platforms

---

## 👥 Contributors

- [Your Name](https://github.com/yourusername)

---

## 📝 License

This project is licensed under the MIT License.

---

## 📚 References

- [Python Docs](https://docs.python.org/3/)
- [Streamlit Tutorial](https://youtu.be/o8p7uQCGD0U)
- [ML Models Overview](https://youtu.be/LvC68w9JS4Y)
- [Dataset Source](https://www.kaggle.com/datasets/varshitanalluri/crop-recommendation-dataset)
