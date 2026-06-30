from __future__ import annotations



import pandas as pd

from fcsl.evaluation.baselines import full_fcsl, observation_only, physical_semantic
from fcsl.evaluation.metrics import evaluate_method


def run_selection_experiment(dataset: pd.DataFrame, epsilon: float = 0.68) -> pd.DataFrame:
    rows = []
    for instance_id, df in dataset.groupby("instance_id"):
        for method, selector in [
            ("Observation-only", observation_only),
            ("Physical-semantic", physical_semantic),
            ("Full FCSL", lambda x: full_fcsl(x, epsilon=epsilon)),
        ]:
            metrics = evaluate_method(df, selector(df), method)
            metrics["instance_id"] = instance_id
            metrics["n_observations"] = int(df["n_observations"].iloc[0])
            rows.append(metrics)
    return pd.DataFrame(rows)



