# app.py
import pickle
import pandas as pd 
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline 
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer

app = Flask(__name__)
CORS(app, origins=[
    "https://seahorse-app-s3y9r.ondigitalocean.app"
  #  "https://mlopsfairnesspipeline.site",
  #  "https://www.mlopsfairnesspipeline.site",
   # "https://seahorse-app-s3y9r.ondigitalocean.app"
])  # to interact as server

# global vars
curr_model = None
curr_scaler = None
curr_baseline = None
curr_preprocessor = None
global_X_fair = None
global_y_fair = None
global_protected_fair= None


# generic config class
class FairnessConfig:
    """ 
    configuration class for relevant fairness-monitoring tasks. User must specify
    datapath, target column(s), and protected attribute(s)
    """
    def __init__(self, df, target, protected_attr, desired=None, privileged=None):
        self.df = df
        self.target = target
        self.protected_attr = protected_attr

        # auto-detect desired & privileged labels (binary 0/1, auto-selects second val alphabetically)
        if desired is None:
            vals = sorted(self.df[target].dropna().unique())
            self.desired = vals[-1]  # ex: 0/1 for >/< 50k
            print(f"auto-detected desired outcome: {self.desired}")
        else:
            self.desired = desired

        if privileged is None:
            vals = sorted(self.df[protected_attr].dropna().unique())
            self.privileged = vals[-1]  # ex: 'male', 'female'
            print(f"auto-detected privileged group: {self.privileged}")
        else:
            self.privileged = privileged

        # auto-detect feature cols, numeric vs categorical
        self.feature_cols = [c for c in self.df.columns if c not in [self.target, self.protected_attr]]
        self.numeric_cols = self.df[self.feature_cols].select_dtypes(include=['int64', 'float64']).columns.tolist()
        self.categorical_cols = self.df[self.feature_cols].select_dtypes(include=['object', 'category', 'string']).columns.tolist()

        print(f"\nauto-detected {len(self.feature_cols)} feature cols, {len(self.numeric_cols)} numeric cols, and {len(self.categorical_cols)} categorical cols")

    def to_dict(self):
        return {
            'target': self.target,
            'protected_attr': self.protected_attr,
            'desired_label': self.desired,
            'privileged_label': self.privileged,
            'feature_cols': self.feature_cols,
            'numeric_cols': self.numeric_cols,
            'categorical_cols': self.categorical_cols }


# helper funcs:
def preprocess(df, target, protected_attr):
    """
    preprocess any relevant dataset using FairnessConfig
    """

    # handle missing vals & create config
    df = df.replace('?', np.nan)
    config = FairnessConfig(df, target, protected_attr)
    df = df.dropna(subset=[config.target, config.protected_attr])
    
    # create binary target & protected attr (1 favorable, 0 unfavorable & 1 privileged, 0 unprivileged)
    y = (df[config.target] == config.desired).astype(int)
    protected_attr = (df[config.protected_attr] == config.privileged).astype(int)
    X = df[config.feature_cols]  # get features
    X.columns = [f'col_{i}' for i in range(len(X.columns))]

    # re-compute numeric and categorical after removing target/protected
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category', 'string']).columns.tolist()
    
    # create preprocessor pipeline
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('encoder', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)])

    X_processed = preprocessor.fit_transform(X)  # fit & transform

    num_cols = numeric_cols
    cat_cols = []
    if len(categorical_cols) > 0:
        cat_encoder = preprocessor.named_transformers_['cat']['encoder']
        cat_cols = cat_encoder.get_feature_names_out(categorical_cols).tolist()

    cols = num_cols + cat_cols
    X_df = pd.DataFrame(X_processed, columns=cols)

    print(f"input features: {X.shape[1]}")
    print(f"output features: {X_df.shape[1]}")
    print(f"target positive: {y.sum()} / {len(y)} ({y.sum()/len(y)*100:.1f}%")
    print(f"privileged group: {protected_attr.sum()} / {len(protected_attr)} ({protected_attr.sum()/len(protected_attr)*100:.1f}%)")
    return X_df.values, y.values, protected_attr.values, preprocessor


def train_model(X, y, protected_attr):
    """
    train model with RandomForest separating data into 80:20 ratio, with:
        - 60% training
        - 20% fairness evaluation
        - 20% testing 
    """

    # separate out fairness set (20%)
    X_temp, X_fair, y_temp, y_fair, protected_temp, protected_fair = train_test_split(
        X, y, protected_attr, test_size = 0.20, random_state = 42, stratify = y)

    # train/test from remaining 80% (60 : 20)
    X_train, X_test, y_train, y_test, protected_train, protected_test = train_test_split(
        X_temp, y_temp, protected_temp,  test_size = 0.25, random_state = 42, stratify = y_temp)

    # balanced set for preservation
    privileged_idx = np.where(protected_fair == 1)[0]
    unprivileged_idx = np.where(protected_fair == 0)[0]

    # set random seed for sampling
    np.random.seed(42)

    # sample equal num for each protected
    min_size = min(len(privileged_idx), len(unprivileged_idx))
    fair_priv_idx = np.random.choice(privileged_idx, min_size, replace = False)
    fair_unpriv_idx = np.random.choice(unprivileged_idx, min_size, replace = False)
    fairness_idx = np.concatenate([fair_priv_idx, fair_unpriv_idx])

    # shuffle idxs
    np.random.shuffle(fairness_idx)

    # create fairness eval set
    X_fair = X_fair[fairness_idx]
    y_fair = y_fair[fairness_idx]
    protected_fair = protected_fair[fairness_idx]

    print(f"Training set size: {X_train.shape[0]} {X_train.shape[0] / len(X) * 100:.1f}%\n")
    print(f"Test size set: {X_test.shape[0]}  {X_test.shape[0] / len(X) * 100:.1f}%\n")
    print(f"Fairness eval set size: {X_fair.shape[0]}  {X_fair.shape[0] / len(X) * 100:.1f}%\n")
    print(f"Protected distributation in fairness set: Privileged = {np.sum(protected_fair)}, Unprivileged = {len(protected_fair) - np.sum(protected_fair)}")

    # random forest classifier initialiized w balanced weights
    model = RandomForestClassifier(
        n_estimators = 200, # tree num
        max_depth = 15, # overfitting
        min_samples_split = 10,
        min_samples_leaf = 5,
        class_weight = 'balanced',
        random_state = 42,
        n_jobs = -1,
        verbose = 1)

    # train model
    model.fit(X_train, y_train)

    print(f"Model trained! Train accuracy: {model.score(X_train, y_train):.3f}, Test accuracy: {model.score(X_test, y_test):.3f}")
    return model, X_fair, y_fair, protected_fair


def calc_fairness(y_true, y_pred, protected, name = ""):
    """
    calculate fairness for all sets
    
    negative - disadvantage for unprivilegeds
    pos - advantage for unprivilegeds
    """
    
    # separate by protected_attr
    unpriv_mask = (protected == 0)
    priv_mask = (protected == 1)

    # pred/true labels
    unpriv_pred = y_pred[unpriv_mask] # unprivileged y 
    unpriv_true = y_true[unpriv_mask]
    priv_pred = y_pred[priv_mask] # privileged y 
    priv_true = y_true[priv_mask]

    metrics = {}

    # stat parity/sel rate difference
    unpriv_pos_rate = unpriv_pred.mean()
    priv_pos_rate = priv_pred.mean()
    stat_parity = unpriv_pos_rate - priv_pos_rate

    metrics['statistical_parity_difference'] = float(stat_parity)
    metrics['unprivileged_sel_rate'] = float(unpriv_pos_rate)
    metrics['privileged_sel_rate'] = float(priv_pos_rate)

    # equal opportunity (true pos rate difference)
    unpriv_pos = unpriv_true.sum()
    priv_pos = priv_true.sum()

    if unpriv_pos > 0 and priv_pos > 0: # determining true pos rate (tpr)
        unpriv_tpr = ((unpriv_pred == 1) & (unpriv_true == 1)).sum() / unpriv_pos
        priv_tpr = ((priv_pred == 1) & (priv_true == 1)).sum() / priv_pos
        eq_op = unpriv_tpr - priv_tpr # equal opportunity 

        metrics['equal_opportunity_difference'] = float(eq_op)
        metrics['unprivileged_tpr'] = float(unpriv_tpr)
        metrics['privileged_tpr'] = float(priv_tpr)

    # predictive parity/precision difference
    unpriv_pred_pos = unpriv_pred.sum() # predicted positive
    priv_pred_pos = priv_pred.sum()

    if unpriv_pred_pos > 0 and priv_pred_pos > 0:
        unpriv_precision = ((unpriv_pred == 1) & (unpriv_true == 1)).sum() / unpriv_pred_pos
        priv_precision = ((priv_pred == 1) & (priv_true == 1)).sum() / priv_pred_pos
        pred_parity = unpriv_precision - priv_precision

        metrics['predictive_parity_difference'] = float(pred_parity)
        metrics['unprivileged_precision'] = float(unpriv_precision)
        metrics['privileged_precision'] = float(priv_precision)

    # accuracy equality
    unpriv_acc = (unpriv_pred == unpriv_true).mean()
    priv_acc = (priv_pred == priv_true).mean()
    acc_diff = unpriv_acc - priv_acc

    metrics['accuracy_difference'] = float(acc_diff)
    metrics['unprivileged_accuracy'] = float(unpriv_acc)
    metrics['privileged_accuracy'] = float(priv_acc)

    return metrics


def inject_bias(predictions, protected, intensity=0.3):
    """
    dataset agnostic bias-injection by flipping predictions 
    """
    predictions = predictions.copy()
    
    # turn unprivileged positives to negatives
    unprivileged_positives = (protected == 0) & (predictions == 1)
    n_flip = int(unprivileged_positives.sum() * intensity)
    
    if n_flip > 0:
        flip_idx = np.random.choice(np.where(unprivileged_positives)[0], n_flip, replace=False)
        predictions[flip_idx] = 0
    
    print(f"Flipped {n_flip} predictions ({intensity*100:.0f}% intensity)")
    return predictions


# API endpoints 
@app.route('/upload', methods=['POST'])
def upload_model():
    """
    loads in csv/.data file, trains model, and returns baseline metrics
    """
    global curr_model, curr_preprocessor, curr_baseline, global_X_fair, global_y_fair, global_protected_fair

    # get file & data
    try:
        file = request.files['dataset']
        target = request.form['target']
        protected_attr = request.form['protected_attr']

        df = pd.read_csv(file)

        # preprocess & train model
        X, y, protected, preprocessor = preprocess(df, target, protected_attr)
        model, X_fair, y_fair, protected_fair = train_model(X, y, protected)

        # get unbiased pred & calc baseline metrics
        y_pred = model.predict(X_fair)
        baseline_metrics = calc_fairness(y_fair, y_pred, protected_fair)
        
        # store globally
        curr_model = model
        curr_preprocessor = preprocessor
        curr_baseline  = baseline_metrics
        global_X_fair = X_fair
        global_y_fair = y_fair
        global_protected_fair = protected_fair

        return jsonify({
            'status': 'success',
            'baseline': {
                'stat_diff': baseline_metrics['statistical_parity_difference'],
                'eq_opp_diff': baseline_metrics['equal_opportunity_difference'],
                'pred_parity_diff': baseline_metrics['predictive_parity_difference'],
                'acc_diff': baseline_metrics['accuracy_difference']
            },
            'message': 'Model train successfully'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/run_batch', methods=['POST'])
def run_batch():
    """
    runs batch with/without bias, returns fairness metrics
    """

    try:
        # check if trained model
        if curr_model is None:
            return jsonify({'status': 'error', 'message': 'No model trained. Upload dataset'}), 400

        # get req data
        data = request.json
        bias = data.get('bias', False)  # check if has bias
        intensity = data.get('intensity', 0)  # check if intensity

        # make predictions
        y_pred = curr_model.predict(global_X_fair)
        
        # inject bias by flipping predictions
        if bias and intensity > 0:
            y_pred = inject_bias(y_pred, global_protected_fair, intensity)
        
        # calc batch metrics
        metrics = calc_fairness(global_y_fair, y_pred, global_protected_fair)

        return jsonify({
            'stat_diff': metrics['statistical_parity_difference'],
            'eq_opp_diff': metrics['equal_opportunity_difference'],
            'pred_parity_diff': metrics['predictive_parity_difference'],
            'acc_diff': metrics['accuracy_difference']
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/health', methods=['GET'])
def check_health():
    """ 
    check server health 
    """
    return jsonify({
        'status': 'running',
        'model': curr_model is not None
    })


# run server
if __name__ == '__main__':
    print("Starting MLOps Fairness backend server...")
    print("Server running on: http://localhost:5000")
    print("Ready to connect to website")
    app.run(host='0.0.0.0', port=5000, debug=False)