# MLOps Dynamic Fairness Monitoring Pipeline
**live demo**: [https://mlopsfairnesspipeline.site](https://mlopsfairnesspipeline.site)

![CI](https://github.com/zarx0130/mlops_fairness_pipeline/workflows/CI%20Pipeline/badge.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)

---
## why?
for the majority of ML models, ethical alignment is not statically set from training data: it remains in a dynamic state that degrades over time, slowly shifting data & causing bias to emerge
- models with high accuracy can still discriminate
- bias often emerges gradually along shifts in data distributions
- most ML models lack real-time ethics monitoring
- early bias detection prevents harm & maintains trust

the consequences of unfair ML models are not abstract, and it can be argued that fairness monitoring should serve as a prerequesite to deployment for all automated systems leading highstakes decisionmaking

--- 

## a solution:
this pipeline provides:

**early detection**: identify bias before it impacts users

**automated scenarios**: simulate different bias patterns to understand model behavior before deployment
  - manual injection with adjustable intensity (0-99%)
  - gradual bias increase/decrease
  - oscillating bias patterns
  - bias detection and correction simulation

**actionable insights**: visualize exactly when and how fairness degrades

**multi-metric fairness analysis**: 
```bash
statistical parity difference
measures difference in positive prediction rates between groups
- formula: P(Ŷ=1|privileged) - P(Ŷ=1|unprivileged)
- fair range: -0.10 to 0.10

equal opportunity difference
measures difference in true positive rates
- formula: TPR(privileged) - TPR(unprivileged)
- fair range: -0.10 to 0.10

predictive parity difference
measures difference in precision
- formula: Precision(privileged) - Precision(unprivileged)
- fair range: -0.10 to 0.10

accuracy equality
measures difference in accuracy
- formula: Accuracy(privileged) - Accuracy(unprivileged)
- fair range: -0.10 to 0.10
```

**real-time batch processing**: monitor model fairness across 25 batches through real-time charts, alerts, and violation tracking

**dataset agnostic**: compatible with binary classification dataset containing headers, a target column, and a protected_attr column

**comprehensive testing**: full test suite with unittest and GitHub Actions CI/CD

---

## installation & application
```bash
# clone repository
git clone https://github.com/zarx0130/mlops_fairness_pipeline.git
cd mlops_fairness_pipeline

# install dependencies
pip install -r requirements.txt
```

---

## local deployment
**start backend:**
```bash
python app.py
```

**open frontend:**
- open `index.html` in your browser
- or visit `http://localhost:5000`

---

## project structure
``` bash
mlops_fairness_pipeline/
├── app.py                  # flask backend
├── index.html              # interactive dashboard
├── requirements.txt        # python dependencies
├── .python-version         # python version (3.11)
├── data/                   # sample datasets
│   ├── adult.data
│   ├── Employee.csv
├── tests/                  # test suite
│   ├── test_backend.py     # unit tests
│   └── test_integration.py # integration tests
└── .github/
└── workflows/
└── ci.yml          # GitHub Actions CI/CD

```

---

## configure dataset:
- click "upload dataset"
- select a CSV file with:
  - binary target column (e.g., `income`, `approved`)
  - binary protected attribute (e.g., `sex`, `race`)
  - additional features
- enter column names
- click "configure & train model"

---

## run batches:
- manual mode: toggle "inject bias", set intensity, click "run batch"
- auto mode: select a scenario, click "begin auto-run"

---

## testing
`python -m unittest discover tests -v`

**umit tests**: backend functions, data preprocessing, model training

**integration tests**: full API workflow w/ datasets

**CI/CD**: automated testing on python 3.9, 3.10, 3.11

---

## testing datasets 
**adult census** (`adult.data`)
- target: `income` (<=50k, >50k)
- protected_attr: `sex` (male, female)

**employee attrition**
- target: `LeaveOrNot` (yes, no)
- protected_attr: `Gender` (male, female)

---

## API endpoints

`POST /upload`
upload dataset and train model

**request:**
```bash
curl -X POST https://mlopsfairnesspipeline.site/upload \
  -F "dataset=@adult.data" \
  -F "target=income" \
  -F "protected_attr=sex"
```

**response:**
```json
{
  "status": "success",
  "baseline": {
    "stat_diff": -0.335,
    "eq_opp_diff": -0.080,
    "pred_parity_diff": 0.014,
    "acc_diff": 0.158
  }
}
```

---

`POST /run_batch`
run batch prediction with optional bias injection

**request:**
```bash
curl -X POST https://mlopsfairnesspipeline.site/run_batch \
  -H "Content-Type: application/json" \
  -d '{"bias": true, "intensity": 0.7}'
```

**response:**
```json
{
  "stat_diff": -0.445,
  "eq_opp_diff": -0.112,
  "pred_parity_diff": 0.089,
  "acc_diff": 0.201
}
```

---

`GET /health`
check backend status

**request:**
```bash
curl https://mlopsfairnesspipeline.site/health
```

**response:**
```json
{
  "status": "running",
  "model": false
}
```

---

## deployment
deployed on Digital Ocean App Platform

1. fork repository
2. create Digital Ocean account
3. create new app → connect GitHub repo
4. configure:
   - build command: `pip install -r requirements.txt`
   - run command: `python app.py`
   - plan: basic ($5/month)
5. deploy

---

## tech stack:
**backend**: flask 3.0.0, scikit-learn 1.3.2, pandas 2.1.4, numpy 1.26.2

**frontend**: HTML5, CSS3, JavaScript (ES6+), Chart.js

**testing**: unittest, coverage, GitHub Actions

---

## license
MIT License - see [LICENSE](LICENSE) file for details

---

**built for responsible AI ♡**