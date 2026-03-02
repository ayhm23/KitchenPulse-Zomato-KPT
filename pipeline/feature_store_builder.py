# pipeline/feature_store_builder.py
"""
KitchenPulse — Feature Store Builder
======================================
Assembles the final feature bundle by wrapping run_denoiser + run_kli.

FIX: No functional changes. Added explicit note that actual_ready_time
     is an EVAL-ONLY column and must not be used as a training feature.
"""

from __future__ import annotations
import pandas as pd
from pipeline.signal_denoiser import run_denoiser
from pipeline.kitchen_load_index import run_kli


class FeatureStoreBuilder:
    def build_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = run_denoiser(df)
        df = run_kli(df)
        df['order_time'] = pd.to_datetime(df['order_time'])
        df['hour'] = df['order_time'].dt.hour
        df['day_of_week'] = df['order_time'].dt.dayofweek
        # EVAL-ONLY columns — do NOT pass to model training:
        #   actual_ready_time, true_kpt_minutes, naive_kpt_error,
        #   actual_rider_wait_minutes, merchant_bias_offset_min
        return df


def build_feature_store(df: pd.DataFrame) -> pd.DataFrame:
    return FeatureStoreBuilder().build_features(df)


if __name__ == '__main__':
    df = pd.read_csv('data/synthetic_orders.csv')
    out = build_feature_store(df)
    print(f"Feature store built. Shape: {out.shape}")
    print(f"Columns: {list(out.columns)}")
