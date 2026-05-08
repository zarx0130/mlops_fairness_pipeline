import sqlite3
import pandas as pd
from datetime import datetime
class FairDB:
    """
    sqlite fairness database creation for prediction logging
    """
    
    def __init__(self, db_path = 'fair.db'):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.create_tables()

    def create_tables(self):
        """
        create tables for individual and batch-level predictions
        """
        
        c = self.conn.cursor()

        # first table for indv. predictions w demographic info
        c.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                model_name TEXT NOT NULL,
                prediction INTEGER NOT NULL,
                true_label INTEGER,
                protected_attr INTEGER NOT NULL,
                batch_id INTEGER,
                time_created DATETIME DEFAULT CURRENT_TIMESTAMP)
                ''')

        # second for batch-level metrics
        c.execute('''
            CREATE TABLE IF NOT EXISTS batch_metrics (
                batch_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                model_name TEXT NOT NULL,
                batch_size INTEGER NOT NULL,
                
                -- stat parirty
                unpriv_sel_rate REAL,
                priv_sel_rate REAL,
                stat_diff REAL,
                
                -- eq op
                unpriv_tpr REAL,
                priv_tpr REAL,
                eq_opp_diff REAL,
                
                -- pred parity
                unpriv_precision REAL,
                priv_precision REAL,
                pred_parity_diff REAL,
               
                -- accuracy
                unpriv_acc REAL,
                priv_acc REAL,
                acc_diff REAL,
                
                time_created DATETIME DEFAULT CURRENT_TIMESTAMP)
                ''')

        # third table for alerts 
        c.execute('''
            CREATE TABLE IF NOT EXISTS alerts (
                alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME NOT NULL,
                batch_id INTEGER NOT NULL,
                metric_name TEXT NOT NULL,
                metric_val REAL NOT NULL,
                threshold REAL NOT NULL,
                severity TEXT NOT NULL,
                message TEXT,
                time_created DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (batch_id) REFERENCES batch_metrics(batch_id))
                ''')

        self.conn.commit()
        print("Database created")

    def log_prediction(self, prediction, protected_attr, model_name, true_label = None, batch_id = None):
        """
        log single predictions
        """
        
        c = self.conn.cursor()
        c.execute('''
            INSERT INTO predictions (timestamp, model_name, prediction, true_label, protected_attr, batch_id)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                datetime.now(),
                model_name,
                int(prediction),
                int(true_label) if true_label is not None else None,
                int(protected_attr),
                batch_id))
        
        self.conn.commit()
        
    def log_batch(self, predictions, protected_attrs, model_name, true_labels = None, batch_id = None):
        """
        log multiple pred at once
        """
        
        c = self.conn.cursor()
        timestamp = datetime.now()

        data = []  # prepare for batch insert
        for i in range(len(predictions)):
            data.append((
                timestamp,
                model_name,
                int(predictions[i]),
                int(true_labels.iloc[i]) if true_labels is not None else None,
                int(protected_attrs.iloc[i] if hasattr(protected_attrs, '__getitem__') else protected_attrs.iloc[i]),
                batch_id))

        c.executemany('''
            INSERT INTO predictions (timestamp, model_name, prediction, true_label, protected_attr, batch_id)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', data)

        self.conn.commit()
        print(f"Logged {len(predictions)} predictions")

    def log_batch_metrics(self, batch_id, model_name, metrics, batch_size):
        """
        log metrics for batch
        """
        
        c = self.conn.cursor()
        c.execute('''
            INSERT INTO batch_metrics (
                batch_id, timestamp, model_name, batch_size, unpriv_sel_rate, priv_sel_rate, stat_diff,
                unpriv_tpr, priv_tpr, eq_opp_diff, unpriv_precision, priv_precision, pred_parity_diff, unpriv_acc,
                priv_acc, acc_diff)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            batch_id, 
            datetime.now(),
            model_name,
            batch_size,
            metrics.get('unprivileged_sel_rate'),
            metrics.get('privileged_sel_rate'),
            metrics.get('statistical_parity_difference'),
            metrics.get('unprivileged_tpr'),
            metrics.get('privileged_tpr'),
            metrics.get('equal_opportunity_difference'),
            metrics.get('unprivileged_precision'),
            metrics.get('privileged_precision'),
            metrics.get('predictive_parity_difference'),
            metrics.get('unprivileged_accuracy'),
            metrics.get('privileged_accuracy'),
            metrics.get('accuracy_difference')
        ))
        self.conn.commit()
        print(f"logged metrics (batch: {batch_id})")

    def log_alert(self, batch_id, metric_name, metric_val, threshold, severity, message=None):
        """
        fairness violation alerts
        """
        
        c = self.conn.cursor()
        c.execute('''
            INSERT INTO alerts
            (timestamp, batch_id, metric_name, metric_val, threshold, severity, message)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.now(), batch_id, metric_name, metric_val, threshold, severity, message))
            
        self.conn.commit()
        print(f"ALERT: {severity} - {metric_name} = {metric_val:.4f}")

    def get_preds(self, limit=100):
        """
        get recent predictions
        """
        
        q = 'SELECT * FROM predictions ORDER BY timestamp DESC LIMIT ?'
        return pd.read_sql_query(q, self.conn, params=(limit,))

    def get_alerts(self, limit=20):
        """
        get recent alerts
        """
        
        q = 'SELECT * FROM alerts ORDER BY timestamp DESC LIMIT ?'
        return pd.read_sql_query(q, self.conn, params=(limit,))

    def get_batch_metrics(self, limit=50):
        """
        get metrics for batch
        """
        
        q = 'SELECT * FROM batch_metrics ORDER BY timestamp DESC LIMIT ?'
        return pd.read_sql_query(q, self.conn, params=(limit,))

    def get_batch_preds(self, batch_id):
        """
        get predictions from specific batch
        """
        
        q = 'SELECT * FROM predictions WHERE batch_id = ?'
        return pd.read_sql_query(q, self.conn, params=(batch_id,))

    def close(self):
        self.conn.close()
        print("db closed")

def calc_batch_fairness(preds_df, threshold=0.05):
    """
    batch metrics calculate fairness
    """
    
    # separate by gender
    unpriv_data = preds_df[preds_df['protected_attr'] == 0]
    priv_data = preds_df[preds_df['protected_attr'] == 1]

    if len(unpriv_data) == 0 or len(priv_data) == 0:
        print(f"Warning: no data for groups")
        return{}

    metrics = {}

    # stat parity
    unpriv_sel = unpriv_data['prediction'].mean()
    priv_sel = priv_data['prediction'].mean()
    stat_diff = unpriv_sel - priv_sel

    metrics['unprivileged_sel_rate'] = float(unpriv_sel)
    metrics['privileged_sel_rate'] = float(priv_sel)
    metrics['statistical_parity_difference'] = float(stat_diff)

    # check if true labels
    labels = 'true_label' in preds_df.columns and preds_df['true_label'].notna().any()
    if labels:
        # equal op
        unpriv_pos = unpriv_data['true_label'].sum()
        priv_pos = priv_data['true_label'].sum()
        
        if unpriv_pos > 0 and priv_pos > 0:
            unpriv_tpr = ((unpriv_data['prediction'] == 1) & (unpriv_data['true_label'] == 1)).sum() / unpriv_pos
            priv_tpr = ((priv_data['prediction'] == 1) & (priv_data['true_label'] == 1)).sum() / priv_pos
            eq_diff = unpriv_tpr - priv_tpr

        metrics['unprivileged_tpr'] = float(unpriv_tpr)
        metrics['privileged_tpr'] = float(priv_tpr)
        metrics['equal_opportunity_difference'] = float(eq_diff)

        # predictive parity
        unpriv_pred_pos = unpriv_data['prediction'].sum()
        priv_pred_pos = priv_data['prediction'].sum()
        if unpriv_pred_pos > 0 and priv_pred_pos > 0:
            unpriv_prec = ((unpriv_data['prediction'] == 1) & (unpriv_data['true_label'] == 1)).sum() / unpriv_pred_pos
            priv_prec = ((priv_data['prediction'] == 1) & (priv_data['true_label'] == 1)).sum() / priv_pred_pos
            prec_diff = unpriv_prec - priv_prec

            metrics['unprivileged_precision'] = float(unpriv_prec)
            metrics['privileged_precision'] = float(priv_prec)
            metrics['predictive_parity_difference'] = float(prec_diff)

        # accuracy equality
        unpriv_acc = (unpriv_data['prediction'] == unpriv_data['true_label']).mean()
        priv_acc = (priv_data['prediction'] == priv_data['true_label']).mean()
        acc_diff = unpriv_acc - priv_acc

        metrics['unprivileged_accuracy'] = float(unpriv_acc)
        metrics['privileged_accuracy'] = float(priv_acc)
        metrics['accuracy_difference'] = float(acc_diff)

    return metrics

def check_violations(metrics, threshold=0.05):
    """
    check which metrics violate fairness threshold
    """
    
    violations = []
    check = [
    ('statistical_parity_difference', 'Statistical Parity'),
    ('equal_opportunity_difference', 'Equal Opportunity'),
    ('predictive_parity_difference', 'Predictive Parity'),
    ('accuracy_difference', 'Accuracy Equality')
    ]

    for key, label in check:
        if key in metrics:
            val = metrics[key]
            if abs(val) > threshold:
                violations.append({
                    'metric_name': key,
                    'label': label,
                    'value': val,
                    'threshold': threshold,
                    'severity': 'critical' if abs(val) > threshold * 2 else 'warning'
                })
    return violations

