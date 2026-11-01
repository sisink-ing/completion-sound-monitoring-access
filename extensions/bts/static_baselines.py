"""Evaluate static predictors on BTS observations."""
from pathlib import Path
import argparse, json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
ROOT = Path(__file__).resolve().parent
IN = ROOT / 'result' / 'prepare_bts'
OUT = ROOT / 'result' / 'static_baselines'
NUM = ['DayOfWeek', 'CRSDepTime', 'CRSElapsedTime', 'Distance']
CAT = ['Reporting_Airline', 'Dest']
FEATURES = NUM + CAT

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--full', action='store_true', help='Use all available online flights instead of first 4600')
    a = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(IN / 'train.csv')
    filename = 'online_full.csv' if a.full else 'online_4600.csv'
    online = pd.read_csv(IN / filename)
    Xtr = train[FEATURES]
    Ytr = train.label.astype(int)
    Xte = online[FEATURES]
    Yte = online.label.astype(int)
    ct = ColumnTransformer([('numeric', Pipeline([('imputer', SimpleImputer(strategy='median')), ('scale', StandardScaler())]), NUM), ('categorical', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', OneHotEncoder(handle_unknown='ignore'))]), CAT)])
    models = {'Static_LR': LogisticRegression(max_iter=1500, random_state=2026), 'Static_DT': DecisionTreeClassifier(max_depth=10, min_samples_leaf=5, random_state=2026)}
    results = {}
    for name, model in models.items():
        pipeline = Pipeline([('transform', ct), ('classifier', model)])
        pipeline.fit(Xtr, Ytr)
        pred = pipeline.predict(Xte)
        metrics = {'online_rows': len(online), 'accuracy': float(accuracy_score(Yte, pred)), 'macro_f1': float(f1_score(Yte, pred, average='macro', zero_division=0)), 'mcc': float(matthews_corrcoef(Yte, pred)), 'predicted_positive_rate': float(pred.mean())}
        results[name] = metrics
        print(name, metrics)
        pd.DataFrame({'event_id': online.event_id, 'historical_prediction': pred, 'true_label_offline_only': Yte}).to_csv(OUT / (name + '_' + filename), index=False)
    (OUT / 'smoke_metrics.json').write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('IMPORTANT: Static smoke tests only; no asynchronous cohort adaptation performed.')
if __name__ == '__main__':
    main()
