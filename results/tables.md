### Synthetic populations, held-out populations, 5 seeds

| method | AUROC | AUROC contextual | AUROC ordinary | AUPRC |
|---|---|---|---|---|
| Isolation Forest | 0.692 ± 0.021 | 0.457 ± 0.049 | 0.929 ± 0.020 | 0.342 ± 0.033 |
| Local Outlier Factor | 0.691 ± 0.020 | 0.463 ± 0.041 | 0.922 ± 0.023 | 0.345 ± 0.033 |
| kNN distance | 0.694 ± 0.021 | 0.467 ± 0.047 | 0.924 ± 0.021 | 0.350 ± 0.035 |
| PCA reconstruction error | 0.583 ± 0.039 | 0.485 ± 0.078 | 0.683 ± 0.026 | 0.197 ± 0.025 |
| Masked MLP, standardized error | 0.692 ± 0.027 | 0.460 ± 0.060 | 0.924 ± 0.018 | 0.322 ± 0.032 |
| Masked MLP, squared error | 0.695 ± 0.021 | 0.459 ± 0.049 | 0.932 ± 0.017 | 0.358 ± 0.032 |
| Context model without context (k=0) | 0.691 ± 0.022 | 0.461 ± 0.045 | 0.924 ± 0.014 | 0.289 ± 0.031 |
| Context model (k=32) | 0.973 ± 0.011 | 0.972 ± 0.014 | 0.974 ± 0.009 | 0.733 ± 0.092 |
| Context model, squared error (k=32) | 0.975 ± 0.009 | 0.969 ± 0.015 | 0.982 ± 0.004 | 0.773 ± 0.066 |
| kNN to the context rows (k=32) | 0.978 ± 0.011 | 0.973 ± 0.020 | 0.983 ± 0.003 | 0.755 ± 0.066 |
| Mahalanobis to the context rows (k=32) | 0.981 ± 0.011 | 0.976 ± 0.019 | 0.984 ± 0.005 | 0.810 ± 0.075 |
| Oracle density of the true population | 1.000 ± 0.000 | 1.000 ± 0.000 | 1.000 ± 0.000 | 0.998 ± 0.001 |

### Synthetic, detectors fit on the rows of the test populations

| method | AUROC | AUROC contextual | AUROC ordinary | AUPRC |
|---|---|---|---|---|
| Isolation Forest, fit on test populations | 0.707 ± 0.022 | 0.471 ± 0.041 | 0.943 ± 0.015 | 0.344 ± 0.033 |
| Local Outlier Factor, fit on test populations | 0.741 ± 0.010 | 0.485 ± 0.014 | 0.995 ± 0.002 | 0.486 ± 0.009 |
| kNN distance, fit on test populations | 0.750 ± 0.039 | 0.506 ± 0.070 | 0.996 ± 0.001 | 0.507 ± 0.025 |
| PCA reconstruction error, fit on test populations | 0.691 ± 0.032 | 0.518 ± 0.049 | 0.865 ± 0.017 | 0.300 ± 0.044 |

### Synthetic, paired differences over seeds

| difference in AUROC | all | contextual | ordinary |
|---|---|---|---|
| context model k=32 minus k=0 | 0.281 ± 0.022 | 0.511 ± 0.040 | 0.049 ± 0.019 |
| context model k=32 minus masked MLP | 0.281 ± 0.025 | 0.512 ± 0.055 | 0.049 ± 0.020 |
| context model k=32 minus Isolation Forest | 0.280 ± 0.020 | 0.515 ± 0.044 | 0.044 ± 0.023 |
| Mahalanobis to context minus context model k=32 | 0.008 ± 0.006 | 0.005 ± 0.009 | 0.011 ± 0.009 |

### Synthetic, anomalies in the context set

| context anomalies | Context model | kNN to context | Mahalanobis to context |
|---|---|---|---|
| 0 % | 0.982 ± 0.008 | 0.980 ± 0.010 | 0.987 ± 0.010 |
| 5 % | 0.971 ± 0.010 | 0.977 ± 0.011 | 0.976 ± 0.013 |
| 10 % | 0.961 ± 0.012 | 0.976 ± 0.012 | 0.970 ± 0.016 |
| 20 % | 0.935 ± 0.018 | 0.971 ± 0.014 | 0.959 ± 0.019 |

### Synthetic, training rows and loss

| training rows and loss | context model k=32 | context model k=32, contextual only | context model k=0 | masked MLP |
|---|---|---|---|---|
| 2 to 5 % anomalies, trimmed loss (default) | 0.973 ± 0.011 | 0.972 ± 0.014 | 0.691 ± 0.022 | 0.692 ± 0.027 |
| 2 to 5 % anomalies, plain loss | 0.899 ± 0.024 | 0.933 ± 0.029 | 0.671 ± 0.024 | 0.651 ± 0.025 |
| no anomalies, plain loss | 0.965 ± 0.017 | 0.974 ± 0.017 | 0.692 ± 0.026 | 0.686 ± 0.025 |

### ADBench tables used

| table | rows | columns | anomalies |
|---|---|---|---|
| breastw | 683 | 9 | 35.0 % |
| cardio | 1,831 | 21 | 9.6 % |
| pima | 768 | 8 | 34.9 % |
| satellite | 6,435 | 36 | 31.6 % |
| shuttle | 49,097 | 9 | 7.2 % |
| thyroid | 3,772 | 6 | 2.5 % |
| vowels | 1,456 | 12 | 3.4 % |
| wbc | 223 | 9 | 4.5 % |

### ADBench, AUROC

| method | breastw | cardio | pima | satellite | shuttle | thyroid | vowels | wbc | mean |
|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest | 0.99 ± 0.01 | 0.93 ± 0.01 | 0.67 ± 0.03 | 0.70 ± 0.02 | 1.00 ± 0.00 | 0.98 ± 0.00 | 0.77 ± 0.02 | 0.99 ± 0.01 | 0.879 |
| LOF | 0.22 ± 0.08 | 0.57 ± 0.04 | 0.62 ± 0.02 | 0.56 ± 0.01 | 0.51 ± 0.02 | 0.67 ± 0.04 | 0.93 ± 0.01 | 0.93 ± 0.03 | 0.627 |
| kNN | 0.97 ± 0.01 | 0.69 ± 0.03 | 0.70 ± 0.03 | 0.66 ± 0.01 | 0.67 ± 0.01 | 0.96 ± 0.00 | 0.98 ± 0.00 | 0.99 ± 0.02 | 0.828 |
| PCA | 0.91 ± 0.03 | 0.75 ± 0.02 | 0.59 ± 0.03 | 0.65 ± 0.01 | 0.74 ± 0.05 | 0.94 ± 0.02 | 0.84 ± 0.04 | 0.96 ± 0.03 | 0.796 |
| Masked MLP, standardized error | 0.52 ± 0.08 | 0.76 ± 0.04 | 0.59 ± 0.05 | 0.59 ± 0.01 | 0.84 ± 0.22 | 0.76 ± 0.05 | 0.85 ± 0.04 | 0.56 ± 0.15 | 0.684 |
| Masked MLP, squared error | 0.96 ± 0.01 | 0.91 ± 0.02 | 0.66 ± 0.05 | 0.70 ± 0.01 | 0.92 ± 0.10 | 0.89 ± 0.03 | 0.94 ± 0.02 | 0.96 ± 0.03 | 0.867 |
| Context model k=0 | 0.53 ± 0.10 | 0.73 ± 0.04 | 0.56 ± 0.05 | 0.57 ± 0.02 | 1.00 ± 0.00 | 0.84 ± 0.09 | 0.83 ± 0.04 | 0.52 ± 0.21 | 0.696 |
| Context model k=32 | 0.55 ± 0.12 | 0.73 ± 0.03 | 0.56 ± 0.05 | 0.57 ± 0.02 | 1.00 ± 0.00 | 0.84 ± 0.09 | 0.82 ± 0.05 | 0.61 ± 0.16 | 0.711 |
| Context model k=32, squared error | 0.97 ± 0.01 | 0.92 ± 0.02 | 0.66 ± 0.04 | 0.71 ± 0.01 | 1.00 ± 0.00 | 0.92 ± 0.01 | 0.92 ± 0.03 | 0.98 ± 0.01 | 0.885 |
| kNN to context | 0.98 ± 0.01 | 0.90 ± 0.02 | 0.71 ± 0.02 | 0.75 ± 0.01 | 0.99 ± 0.00 | 0.97 ± 0.01 | 0.86 ± 0.02 | 0.99 ± 0.01 | 0.893 |
| Mahalanobis to context | 0.97 ± 0.01 | 0.89 ± 0.02 | 0.66 ± 0.02 | 0.64 ± 0.01 | 0.98 ± 0.00 | 0.95 ± 0.01 | 0.84 ± 0.04 | 0.97 ± 0.02 | 0.864 |

### ADBench, AUPRC

| method | breastw | cardio | pima | satellite | shuttle | thyroid | vowels | wbc | mean |
|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest | 0.97 ± 0.01 | 0.60 ± 0.06 | 0.51 ± 0.06 | 0.66 ± 0.00 | 0.98 ± 0.01 | 0.66 ± 0.06 | 0.22 ± 0.10 | 0.92 ± 0.11 | 0.691 |
| LOF | 0.25 ± 0.01 | 0.19 ± 0.04 | 0.44 ± 0.02 | 0.38 ± 0.02 | 0.12 ± 0.01 | 0.04 ± 0.01 | 0.38 ± 0.07 | 0.32 ± 0.09 | 0.267 |
| kNN | 0.93 ± 0.03 | 0.32 ± 0.05 | 0.54 ± 0.05 | 0.51 ± 0.01 | 0.18 ± 0.01 | 0.28 ± 0.03 | 0.69 ± 0.07 | 0.86 ± 0.13 | 0.537 |
| PCA | 0.85 ± 0.04 | 0.34 ± 0.03 | 0.44 ± 0.03 | 0.51 ± 0.01 | 0.19 ± 0.02 | 0.56 ± 0.08 | 0.23 ± 0.10 | 0.58 ± 0.19 | 0.464 |
| Masked MLP, standardized error | 0.36 ± 0.04 | 0.54 ± 0.05 | 0.44 ± 0.04 | 0.46 ± 0.02 | 0.66 ± 0.40 | 0.12 ± 0.07 | 0.33 ± 0.13 | 0.10 ± 0.09 | 0.376 |
| Masked MLP, squared error | 0.92 ± 0.03 | 0.54 ± 0.10 | 0.49 ± 0.06 | 0.57 ± 0.01 | 0.71 ± 0.29 | 0.44 ± 0.08 | 0.54 ± 0.13 | 0.63 ± 0.16 | 0.603 |
| Context model k=0 | 0.41 ± 0.12 | 0.51 ± 0.04 | 0.42 ± 0.05 | 0.48 ± 0.02 | 0.95 ± 0.01 | 0.24 ± 0.07 | 0.35 ± 0.16 | 0.21 ± 0.23 | 0.448 |
| Context model k=32 | 0.42 ± 0.14 | 0.53 ± 0.04 | 0.42 ± 0.05 | 0.49 ± 0.02 | 0.96 ± 0.01 | 0.23 ± 0.07 | 0.33 ± 0.15 | 0.22 ± 0.23 | 0.450 |
| Context model k=32, squared error | 0.92 ± 0.03 | 0.59 ± 0.09 | 0.49 ± 0.06 | 0.60 ± 0.01 | 0.93 ± 0.01 | 0.46 ± 0.08 | 0.42 ± 0.14 | 0.74 ± 0.09 | 0.643 |
| kNN to context | 0.95 ± 0.02 | 0.56 ± 0.09 | 0.52 ± 0.04 | 0.67 ± 0.01 | 0.91 ± 0.01 | 0.45 ± 0.06 | 0.20 ± 0.02 | 0.88 ± 0.08 | 0.644 |
| Mahalanobis to context | 0.94 ± 0.02 | 0.48 ± 0.07 | 0.50 ± 0.05 | 0.55 ± 0.01 | 0.78 ± 0.01 | 0.46 ± 0.04 | 0.20 ± 0.05 | 0.71 ± 0.16 | 0.578 |

### ADBench, paired differences in AUROC over tables and splits

| paired difference in AUROC | mean | 95 % interval | tables above 0 | splits above 0 |
|---|---|---|---|---|
| standardized error: context k=32 minus k=0 | +0.015 | [-0.001, +0.040] | 7 of 8 | 31 of 40 |
| standardized error: context k=32 minus masked MLP | +0.027 | [-0.017, +0.083] | 4 of 8 | 17 of 40 |
| standardized error: context k=32 minus kNN to context | -0.182 | [-0.287, -0.088] | 1 of 8 | 6 of 40 |
| standardized error: context k=32 minus Isolation Forest | -0.169 | [-0.288, -0.063] | 2 of 8 | 7 of 40 |
| squared error: context k=32 minus masked MLP | +0.018 | [-0.001, +0.042] | 6 of 8 | 30 of 40 |
| squared error: context k=32 minus kNN to context | -0.008 | [-0.031, +0.017] | 3 of 8 | 16 of 40 |
| squared error: context k=32 minus Isolation Forest | +0.006 | [-0.028, +0.053] | 3 of 8 | 19 of 40 |
| context k=32: squared minus standardized error | +0.174 | [+0.086, +0.278] | 7 of 8 | 36 of 40 |

### ADBench, test rows with an exact copy in the training rows

| table | rows | rows in a repeated vector | plain split: copy in the training rows | plain split: copy in a scored context | grouped split: any copy |
|---|---|---|---|---|---|
| breastw | 683 | 280 | 390 of 665 / 6 of 360 | 168 of 665 / 0 of 360 | 0 |
| cardio | 1,831 | 16 | 16 of 2,485 / 2 of 265 | 0 of 2,485 / 0 of 265 | 0 |
| pima | 768 | 0 | 0 of 750 / 0 of 405 | 0 of 750 / 0 of 405 | 0 |
| satellite | 6,435 | 0 | 0 of 6,600 / 0 of 3,055 | 0 of 6,600 / 0 of 3,055 | 0 |
| shuttle | 49,097 | 0 | 0 of 68,385 / 0 of 5,265 | 0 of 68,385 / 0 of 5,265 | 0 |
| thyroid | 3,772 | 163 | 211 of 5,520 / 0 of 140 | 4 of 5,520 / 0 of 140 | 0 |
| vowels | 1,456 | 8 | 0 of 2,110 / 10 of 75 | 0 of 2,110 / 1 of 75 | 0 |
| wbc | 223 | 0 | 0 of 320 / 0 of 15 | 0 of 320 / 0 of 15 | 0 |

### AUROC against context size

| k | Context model | kNN to context | Mahalanobis to context | ADBench mean, context model |
|---|---|---|---|---|
| 0 | 0.691 ± 0.022 | n/a | n/a | 0.696 |
| 4 | 0.926 ± 0.020 | 0.916 ± 0.015 | 0.896 ± 0.019 | 0.711 |
| 8 | 0.951 ± 0.016 | 0.949 ± 0.014 | 0.943 ± 0.019 | 0.711 |
| 16 | 0.965 ± 0.015 | 0.966 ± 0.013 | 0.967 ± 0.015 | 0.711 |
| 32 | 0.973 ± 0.011 | 0.978 ± 0.011 | 0.981 ± 0.011 | 0.711 |
| 64 | 0.979 ± 0.005 | 0.986 ± 0.007 | 0.987 ± 0.008 | 0.711 |

### Schema change, half of the columns

| method | synthetic, all columns | synthetic, half | ADBench, all | ADBench, half |
|---|---|---|---|---|
| Isolation Forest | 0.692 ± 0.021 | 0.671 ± 0.012 | 0.879 | 0.846 |
| Local Outlier Factor | 0.691 ± 0.020 | 0.667 ± 0.012 | 0.627 | 0.650 |
| kNN distance | 0.694 ± 0.021 | 0.666 ± 0.013 | 0.828 | 0.805 |
| PCA reconstruction error | 0.583 ± 0.039 | 0.596 ± 0.022 | 0.796 | 0.686 |
| Masked MLP, standardized error | 0.692 ± 0.027 | 0.671 ± 0.015 | 0.684 | 0.735 |
| Masked MLP, squared error | 0.695 ± 0.021 | 0.671 ± 0.014 | 0.867 | 0.832 |
| Context model without context (k=0) | 0.691 ± 0.022 | 0.670 ± 0.014 | 0.696 | 0.720 |
| Context model (k=32) | 0.973 ± 0.011 | 0.925 ± 0.011 | 0.711 | 0.730 |
| Context model, squared error (k=32) | 0.975 ± 0.009 | 0.926 ± 0.010 | 0.885 | 0.841 |
| kNN to the context rows (k=32) | 0.978 ± 0.011 | 0.942 ± 0.014 | 0.893 | 0.858 |
| Mahalanobis to the context rows (k=32) | 0.981 ± 0.011 | 0.935 ± 0.016 | 0.864 | 0.825 |
| Oracle density of the true population | 1.000 ± 0.000 | 0.980 ± 0.005 | n/a | n/a |

### Training cost

| run | context model s | masked MLP s | parameters (context, MLP) |
|---|---|---|---|
| synthetic, one seed | 147 | 99 | 146,452, 142,100 |
| ADBench, both models, one table | 92 |  |  |
