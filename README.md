# MLOps Dynamic Fairness Monitoring Pipeline
prototyping real-time automated bias detection through python, a Flask-based API integration, and interpretable interactive dashboard

## why?
for the majority of ML models, ethical alignment is not statically set from training data: it remains in a dynamic state that degrades over time, slowly shifting data & causing bias to emerge
- models with high accuracy can still discriminate
- bias often emerges gradually along shifts in data distributions
- most ML models lack real-time ethics monitoring
- early bias detection prevents harm & maintains trust

the consequences of unfair ML models are not abstract, and it can be argued that fairness monitoring should serve as a prerequesite to deployment for all automated systems leading highstakes decisionmaking

## a solution:
this pipeline provides:

**early detection**: identify bias before it impacts users

**automated scenarios**: simulate different bias patterns to understand model behavior before deployment
  - manual injection with adjustable intensity (0-99%)
  - gradual bias increase/decrease
  - oscillating bias patterns
  - bias detection and correction simulation

**actionable insights**: visualize exactly when and how fairness degrades

**multi-metric fairness analysis**: statistical parity, equal opportunity, predictive parity, accuracy

**real-time batch processing**: monitor model fairness across 25 batches through real-time charts, alerts, and violation tracking

**dataset agnostic**: compatible with binary classification dataset containing headers, a target column, and a protected_attr column

**comprehensive testing**: full test suite with unittest and GitHub Actions CI/CD

## installation & application
**clone repository**
`git clone https://github.com/zarx0130/mlops_fairness_pipeline.git`

`cd mlops_fairness_pipeline`

**install dependencies**
pip install -r requirements.txt




**configure dataset:**
- click "upload dataset"
- select a CSV file with:
  - binary target column (e.g., `income`, `approved`)
  - binary protected attribute (e.g., `sex`, `race`)
  - additional features
- enter column names
- click "configure & train model"

**run batches:**
- manual mode: toggle "inject bias", set intensity, click "run batch"
- auto mode: select a scenario, click "begin auto-run"

## testing
`python -m unittest discover tests -v`

**umit tests**: backend functions, data preprocessing, model training

**integration tests**: full API workflow w/ datasets

**CI/CD**: automated testing on python 3.9, 3.10, 3.11

## testing datasets 
**adult census** (`adult.data`)
- target: `income` (<=50k, >50k)
- protected_attr: `sex` (male, female)

**employee attrition**
- target: `LeaveOrNot` (yes, no)
- protected_attr: `Gender` (male, female)

## API endpoints
 `POST /upload` - upload dataset & train model
 
 `POST /run_batch` - run batch prediction w/ optional bias injection
 
 `GET /health` - check backend status

## deployment - digital ocean
