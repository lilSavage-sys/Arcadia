import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class FactorScorer:
    def __init__(self, factor_config):
        self.factor_config = factor_config

    def normalize(self, df, method='zscore'):
        if method == 'zscore':
            scaler = StandardScaler()
            return pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
        elif method == 'percentile':
            return df.rank(pct=True)
        else:
            raise ValueError('Unknown normalization method')

    def score(self, data):
        # data: DataFrame with columns for each factor
        scores = {}
        for group, info in self.factor_config.items():
            metrics = info['metrics']
            weight = info['weight']
            group_df = data[metrics]
            normed = self.normalize(group_df)
            scores[group] = normed.mean(axis=1) * weight
        composite = sum(scores.values())
        return composite, scores
