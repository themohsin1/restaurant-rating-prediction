
import pandas as pd
import numpy as np
from collections import Counter

class FeatureBuilder:
    def __init__(self, topk=30):
        self.topk = topk
        self.loc_freq_ = None
        self.type_freq_ = None
        self.cuisine_vocab_ = None

    def fit(self, df):
        self.loc_freq_  = df["location"].value_counts(normalize=True)
        self.type_freq_ = df["rest_type"].value_counts(normalize=True)
        def split(s): return [x.strip() for x in str(s).split(",") if x.strip()]
        counts = Counter(c for row in df["cuisines"].map(split) for c in row)
        self.cuisine_vocab_ = [c for c,_ in counts.most_common(self.topk)]
        return self

    def transform(self, df):
        out = pd.DataFrame(index=df.index)
        out["online_order"] = df["online_order"].map({"yes":1,"no":0}).fillna(0)
        out["book_table"]   = df["book_table"].map({"yes":1,"no":0}).fillna(0)
        out["location_freq"] = df["location"].map(self.loc_freq_).fillna(0)
        out["rest_type_freq"] = df["rest_type"].map(self.type_freq_).fillna(0)
        # Clean the 'approx_cost(for two people)' column: remove commas and convert to float
        out["cost_log"] = np.log1p(
            df["approx_cost(for two people)"]
            .astype(str)
            .str.replace(",", "")
            .astype(float)
        )


        def split(s): return [x.strip() for x in str(s).split(",") if x.strip()]
        for cu in self.cuisine_vocab_:
            out[f"cuisine__{cu}"] = df["cuisines"].map(lambda r: int(cu in split(r)))
        return out
