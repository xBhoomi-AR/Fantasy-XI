"""Candidate generation / PPO / MILP decision layer for Fantasy XI.

Only consumes models/xgboost_model's (and later BiLSTM's) output files, not
their internals, so the prediction source can be swapped later.
"""
