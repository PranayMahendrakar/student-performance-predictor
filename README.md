# Student Performance Predictor

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32-red?logo=streamlit)](https://streamlit.io)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4-orange)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A machine-learning web application that predicts student academic performance (Pass / At Risk / Fail) from engagement and assessment metrics, and generates personalised improvement suggestions.

---

## Features

- **Dual-model prediction** — Random Forest & Gradient Boosting (sklearn)
- **Ensemble mode** — averages both models for higher reliability
- **8 input features** — attendance, study hours, assignments, quizzes, GPA, etc.
- **Personalised suggestions** — ranked by feature importance & severity
- **Batch analysis** — upload a CSV of students and get a full cohort report
- **Interactive visualisations** — probability bars, radar chart, correlation heatmap, box-plots
- **3-class output** — Pass | At Risk | Fail

---

## Project Structure

```
student-performance-predictor/
|- app.py                   # Streamlit UI (3 tabs: Predict, Batch, Insights)
|- requirements.txt         # Python dependencies
|- model/
|  |- train_model.py        # Dataset generation + RF & GBM training
|  |- predictor.py          # Inference engine + suggestion generator
|- artifacts/               # Saved models (generated after training)
|  |- random_forest.pkl
|  |- gradient_boosting.pkl
|  |- scaler.pkl
|  |- label_encoder.pkl
|  |- model_metadata.json
|- data/
|  |- student_data.csv      # Sample dataset (generated after training)
|- LICENSE
|- README.md
```

---

## Quick Start

### 1. Clone & install

```bash
git clone https://github.com/PranayMahendrakar/student-performance-predictor.git
cd student-performance-predictor
pip install -r requirements.txt
```

### 2. Train the models

```bash
python model/train_model.py
```

This generates `artifacts/` with trained models and `data/student_data.csv`.

### 3. Launch the app

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## Input Features

| Feature | Range | Description |
|---|---|---|
| `attendance_rate` | 0 – 100 % | % of classes attended |
| `assignment_avg` | 0 – 100 % | Mean assignment score |
| `midterm_score` | 0 – 100 % | Midterm exam score |
| `study_hours_per_week` | 0 – 20 hrs | Self-reported study time |
| `participation_score` | 0 – 10 | Classroom participation |
| `previous_gpa` | 0.0 – 4.0 | Prior semester GPA |
| `assignment_completion` | 0 – 100 % | % of assignments submitted |
| `quiz_avg` | 0 – 100 % | Mean quiz score |

---

## Output

### Performance Prediction

| Label | Meaning |
|---|---|
| Pass | Student is performing well |
| At Risk | Intervention may be needed |
| Fail | Immediate support required |

### Improvement Suggestions

The app generates up to 6 actionable suggestions ranked by:

1. Feature importance (from the chosen model)
2. How far the student is below the threshold

Example suggestions include:
- "Boost attendance to at least 75%…"
- "Aim for 10+ study hours per week…"
- "Submit at least 85% of assignments…"

---

## Models

### Random Forest

- **Estimators:** 200 trees
- **Max depth:** 10
- **Class weight:** balanced
- **Advantages:** robust to noise, easy feature importance interpretation

### Gradient Boosting

- **Estimators:** 200
- **Learning rate:** 0.05
- **Max depth:** 5
- **Subsample:** 0.8
- **Advantages:** typically higher accuracy, captures complex patterns

### Ensemble

Averages the probability distributions of both models for a more stable final prediction.

---

## Batch Analysis

Upload a CSV file with the 8 feature columns to predict performance for a whole class:

```csv
attendance_rate,assignment_avg,midterm_score,study_hours_per_week,participation_score,previous_gpa,assignment_completion,quiz_avg
85,78,72,10,7,3.2,90,75
45,52,48,4,3,1.8,60,50
```

The app outputs:
- Pass / At Risk / Fail counts
- Pie chart of distribution
- Scatter plot: attendance vs study hours
- Full tabular results

---

## Tech Stack

- **Frontend:** Streamlit 1.32
- **ML:** scikit-learn 1.4 (RandomForestClassifier, GradientBoostingClassifier)
- **Data:** pandas, numpy
- **Visualisation:** Plotly Express & Graph Objects
- **Persistence:** joblib

---

## License

MIT License — see [LICENSE](LICENSE) for details.

