import pandas as pd

def rank_stocks(composite_scores, top_n=20):
    ranked = composite_scores.sort_values(ascending=False)
    return ranked.head(top_n)
