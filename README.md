# contextad

When what counts as normal depends on the population a row comes from, does an anomaly detector that sees a context set (other rows of the same population) beat detectors that score rows in isolation, and does it keep working when the set of columns changes?

The study runs on a laptop CPU: a synthetic generator with contextual anomalies, eight small ADBench tables, scikit-learn baselines, a masked MLP, and a small attention model over a context set.
