# 🛍️ Fashion Forward Forecasting - StyleSense Product Recommendation Pipeline

> **Build a machine learning pipeline to predict customer product recommendations using NLP and scikit-learn**

[![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE.txt)
[![Status](https://img.shields.io/badge/Status-Complete-success.svg)]()

---

## 📌 Project Overview

This is a **complete machine learning project** demonstrating a production-ready pipeline for predicting product recommendations. The project handles mixed data types (numerical, categorical, text) and implements best practices including data preprocessing, feature engineering, model training, cross-validation, and hyperparameter tuning.

### 🎯 Problem Statement

StyleSense, an online women's clothing retailer, receives thousands of customer reviews but not all include explicit recommendation indicators. This project automates recommendation prediction by analyzing review text, customer demographics, and product information.

### 📊 Dataset

- **18,442** customer reviews
- **8 features** (numerical, categorical, text)
- **81.6%** positive class distribution
- **9 columns** total (including target)

---

## 🚀 Quick Start

### ⚡ 5-Minute Setup

```bash
# Clone repository
git clone https://github.com/udacity/dsnd-pipelines-project.git
cd dsnd-pipelines-project

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy model
python -m spacy download en_core_web_sm

# Start Jupyter
jupyter notebook
```

Then open **`starter/starter.ipynb`** and follow the cells!

---

## 📦 Dependencies

### Core Libraries
```
scikit-learn>=1.0.0      # Machine Learning
pandas>=1.3.0            # Data Manipulation
numpy>=1.21.0            # Numerical Computing
spacy>=3.0.0             # NLP Processing
```

### Visualization & Jupyter
```
notebook>=6.4.0          # Jupyter Notebook
matplotlib>=3.4.0        # Plotting
seaborn>=0.11.0          # Statistical Plots
```

### Installation Methods

**Option 1: Using requirements.txt**
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

**Option 2: Manual Installation**
```bash
pip install scikit-learn pandas numpy spacy notebook matplotlib seaborn
python -m spacy download en_core_web_sm
```

**For Mac M1/M2:**
```bash
pip install 'spacy[apple]'
python -m spacy download en_core_web_sm
```

---

## 📁 Project Structure

```
dsnd-pipelines-project/
│
├── starter/                          # 📂 Student-facing folder
│   ├── README.md                     # ✍️ Instructions for students
│   ├── starter.ipynb                 # 📓 Main notebook template
│   └── data/
│       └── reviews.csv               # 📊 Dataset (18,442 reviews)
│
├── requirements.txt                  # 📝 Project dependencies
├── README.md                         # 📖 This file
├── LICENSE.txt                       # ⚖️ MIT License
├── .gitignore                        # 🔒 Git ignore rules
└── CODEOWNERS                        # 👥 Project maintainers
```

### 📂 Key Folders

| Folder | Purpose |
|--------|---------|
| **`starter/`** | Scaffolded project files for students |
| **`starter/data/`** | Customer review CSV dataset |

---

## 📋 Project Instructions

### Part 1️⃣: Data Exploration

Understand the dataset by examining:
- Data types and distributions
- Missing values
- Feature statistics
- Text data characteristics
- Class balance

### Part 2️⃣: Feature Engineering

Create new features from raw data:
- **Text Features**: Review length, word count, sentiment indicators
- **Numerical Features**: Already provided (age, feedback count)
- **Categorical Features**: Product divisions, departments, classes

### Part 3️⃣: Build Pipeline

Construct a complete ML pipeline:

```python
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features),
        ('text', TfidfVectorizer(), text_features)
    ])),
    ('classifier', RandomForestClassifier(n_estimators=100))
])
```

### Part 4️⃣: Train Model

- Train on 16,597 samples (90%)
- Evaluate on training set
- Perform 5-fold cross-validation

### Part 5️⃣: Fine-Tune Hyperparameters

- Use GridSearchCV with parameter grid
- Test 36 parameter combinations
- Select best model based on F1-score

### Part 6️⃣: Evaluate on Test Set

- Test on 1,845 samples (10%)
- Report accuracy, precision, recall, F1-score
- Analyze confusion matrix
- Visualize results

---

## 📊 Expected Results

When properly implemented, the model should achieve:

```
╔═══════════════════════════════════════╗
║      EXPECTED MODEL PERFORMANCE       ║
╠═══════════════════════════════════════╣
║  Test Accuracy:   ~84-85%            ║
║  Test Precision:  ~85%               ║
║  Test Recall:     ~98-99%            ║
║  Test F1-Score:   ~91%               ║
╚═══════════════════════════════════════╝
```

### Key Metrics Explained

| Metric | Interpretation |
|--------|----------------|
| **Accuracy** | Overall correctness of predictions |
| **Precision** | Of predicted recommendations, how many correct |
| **Recall** | Of actual recommendations, how many found |
| **F1-Score** | Harmonic mean (balanced precision/recall) |

---

## 🧠 NLP Techniques

This project demonstrates several NLP preprocessing techniques:

1. **🔤 Tokenization** - Split text into words/tokens
2. **🚫 Stop Word Removal** - Filter common words (the, a, and)
3. **2️⃣ N-gram Creation** - Extract word pairs and single words
4. **📈 TF-IDF Vectorization** - Convert text to numerical features
5. **😊 Sentiment Analysis** - Count positive/negative words
6. **📏 Feature Scaling** - Normalize numerical values

---

## 🏗️ Pipeline Architecture

```
Raw Data (18,442 reviews)
    ↓
┌──────────────────────────────┐
│  FEATURE ENGINEERING         │
│  • Text features             │
│  • Sentiment indicators      │
└──────────────────────────────┘
    ↓
┌──────────────────────────────┐
│  PREPROCESSING               │
│  • StandardScaler (numeric)  │
│  • OneHotEncoder (category)  │
│  • TfidfVectorizer (text)    │
└──────────────────────────────┘
    ↓
┌──────────────────────────────┐
│  FEATURE MATRIX              │
│  (~162 total features)       │
└──────────────────────────────┘
    ↓
┌──────────────────────────────┐
│  RANDOM FOREST CLASSIFIER    │
│  (200 decision trees)        │
└──────────────────────────────┘
    ↓
🎯 Binary Prediction (0 or 1)
```

---

## 📖 How to Run

### Step 1: Navigate to Project

```bash
cd dsnd-pipelines-project
```

### Step 2: Activate Environment

```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

### Step 3: Start Jupyter

```bash
jupyter notebook
```

### Step 4: Open Notebook

Click on **`starter/starter.ipynb`**

### Step 5: Run Cells Sequentially

Execute each cell from top to bottom. Each cell builds on previous results.

---

## 🧪 Testing

The project includes integrated testing:

- ✅ **Train-Test Split**: 90-10 split prevents data leakage
- ✅ **Cross-Validation**: 5-fold CV validates generalization
- ✅ **Multiple Metrics**: Accuracy, precision, recall, F1-score
- ✅ **Confusion Matrix**: Detailed error analysis
- ✅ **Visualizations**: Plots for results interpretation

**No external test suite required** - evaluation is built into the notebook.

---

## 🐛 Troubleshooting

### Issue: ModuleNotFoundError: No module named 'spacy'

**Solution:**
```bash
pip install spacy
python -m spacy download en_core_web_sm
```

### Issue: Data file not found

**Solution:**
Ensure you're running the notebook from `starter/` folder or adjust the path:
```python
df = pd.read_csv('data/reviews.csv')  # From starter/ folder
```

### Issue: GridSearchCV taking too long

**Solution:**
- Reduce CV folds: `cv=3` instead of `cv=5`
- Reduce parameter combinations
- Use smaller dataset for testing

### Issue: Out of memory error

**Solution:**
```python
# Reduce features in TfidfVectorizer
TfidfVectorizer(max_features=50)  # Reduce from 100

# Reduce categories in OneHotEncoder
OneHotEncoder(max_categories=20)  # Reduce from 50
```

---

## ✨ Features & Highlights

- ✅ **Complete pipeline** from data to predictions
- ✅ **Handles mixed data types** (numerical, categorical, text)
- ✅ **Proper train/test split** with no data leakage
- ✅ **Cross-validation** for robust evaluation
- ✅ **Hyperparameter tuning** with GridSearchCV
- ✅ **NLP preprocessing** with multiple techniques
- ✅ **Multiple evaluation metrics** for comprehensive assessment
- ✅ **Well-documented code** with docstrings
- ✅ **Professional visualizations** of results

---

## 🚀 Advanced Usage

### Save Trained Model

```python
import joblib

# Save the pipeline
joblib.dump(best_pipeline, 'recommendation_model.pkl')

# Load the pipeline
loaded_pipeline = joblib.load('recommendation_model.pkl')
```

### Make Batch Predictions

```python
# On new data
predictions = best_pipeline.predict(X_new)
probabilities = best_pipeline.predict_proba(X_new)

# Create results dataframe
results = pd.DataFrame({
    'prediction': predictions,
    'probability': probabilities[:, 1]
})
```

### Extract Feature Importance

```python
# Get feature importance from Random Forest
importance = best_pipeline.named_steps['classifier'].feature_importances_

# Display top features
top_features = pd.Series(importance).nlargest(10)
print(top_features)
```

---

## 📚 Learning Resources

| Resource | Link |
|----------|------|
| **scikit-learn Docs** | https://scikit-learn.org/stable/ |
| **pandas Docs** | https://pandas.pydata.org/docs/ |
| **Jupyter Guide** | https://jupyter.readthedocs.io/ |
| **spaCy Tutorial** | https://spacy.io/usage |
| **TF-IDF Explanation** | https://en.wikipedia.org/wiki/Tf%E2%80%93idf |

---

## ⚖️ License

This project is licensed under the **MIT License** - see [LICENSE.txt](LICENSE.txt) for details.

### You are free to:
- ✅ Use this project
- ✅ Modify the code
- ✅ Distribute copies
- ✅ Use it commercially

### Conditions:
- ⚠️ Include license and copyright notice

---

## 🛠️ Built With

| Technology | Purpose |
|-----------|---------|
| 🐍 **Python 3.7+** | Programming language |
| 🤖 **scikit-learn** | ML algorithms & preprocessing |
| 📊 **pandas** | Data manipulation |
| 🔢 **NumPy** | Numerical computing |
| 🧠 **spaCy** | NLP text processing |
| 📓 **Jupyter** | Interactive notebooks |
| 📈 **Matplotlib** | Data visualization |
| 🎨 **Seaborn** | Statistical plots |

---

## 👥 Contributing

This is an **educational project** for Udacity's Data Science Nanodegree.

For improvements or bug reports, please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📞 Support

### Getting Help

1. **Check the starter/README.md** for student instructions
2. **Read code comments** in starter.ipynb
3. **Review the rubric** for requirements
4. **Google the error** with library name
5. **Check documentation** of relevant libraries

### Common Questions

**Q: Why am I getting different results?**
A: Random state should be 27 in train_test_split and RandomForest.

**Q: What if my accuracy is too low?**
A: Check data preprocessing, feature engineering, and model parameters.

**Q: Can I use a different classifier?**
A: Yes! Try XGBoost, GradientBoosting, or SVM for comparison.

---

## 🎓 Educational Value

This project teaches:
- 🔄 End-to-end ML pipeline development
- 📝 Data preprocessing and feature engineering
- 🧠 NLP text processing techniques
- 🎯 Model training and evaluation
- ⚙️ Hyperparameter optimization
- 📊 Cross-validation and metrics
- 📈 Results visualization and interpretation

---

## 📈 Performance Summary

| Component | Status | Details |
|-----------|--------|---------|
| **Data Loading** | ✅ Complete | 18,442 reviews loaded |
| **Preprocessing** | ✅ Complete | All 3 data types handled |
| **Feature Engineering** | ✅ Complete | 8 new features created |
| **Model Pipeline** | ✅ Complete | Integrated preprocessing + classifier |
| **Training** | ✅ Complete | 16,597 samples trained |
| **Evaluation** | ✅ Complete | Multiple metrics reported |
| **Tuning** | ✅ Complete | GridSearchCV with 36 combos |
| **Testing** | ✅ Complete | 1,845 samples evaluated |

---

## 🎉 Next Steps

After completing this project:

1. 🚀 Deploy the model as an API
2. 📊 Build a dashboard for predictions
3. 📈 Try advanced ML techniques
4. 🔄 Implement continuous monitoring
5. 💡 Apply to your own datasets

---

## 📝 Citation

If you use this project, please cite:

```bibtex
@project{StyleSenseRecommendation,
  title={Fashion Forward Forecasting: StyleSense Product Recommendation Pipeline},
  author={Udacity Data Science Team},
  year={2024},
  url={https://github.com/udacity/dsnd-pipelines-project}
}
```

---

**Made with ❤️ for aspiring data scientists**

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black.svg)](https://github.com/udacity/dsnd-pipelines-project)
[![Udacity](https://img.shields.io/badge/Udacity-Data%20Science%20ND-blue.svg)](https://www.udacity.com/)

**Last Updated**: 2024 | **Status**: Production Ready ✨
