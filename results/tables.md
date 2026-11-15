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
| Isolation Forest | 0.99 ± 0.00 | 0.93 ± 0.01 | 0.67 ± 0.03 | 0.70 ± 0.02 | 1.00 ± 0.00 | 0.98 ± 0.00 | 0.76 ± 0.05 | 0.99 ± 0.01 | 0.877 |
| LOF | 0.49 ± 0.05 | 0.58 ± 0.03 | 0.62 ± 0.02 | 0.56 ± 0.01 | 0.51 ± 0.02 | 0.67 ± 0.03 | 0.93 ± 0.01 | 0.93 ± 0.03 | 0.661 |
| kNN | 0.98 ± 0.01 | 0.69 ± 0.03 | 0.70 ± 0.03 | 0.66 ± 0.01 | 0.67 ± 0.01 | 0.96 ± 0.00 | 0.97 ± 0.01 | 0.99 ± 0.02 | 0.828 |
| PCA | 0.92 ± 0.02 | 0.75 ± 0.02 | 0.59 ± 0.03 | 0.65 ± 0.01 | 0.74 ± 0.05 | 0.94 ± 0.02 | 0.84 ± 0.03 | 0.96 ± 0.03 | 0.798 |
| Masked MLP, standardized error | 0.58 ± 0.01 | 0.75 ± 0.03 | 0.59 ± 0.05 | 0.59 ± 0.01 | 0.84 ± 0.22 | 0.79 ± 0.02 | 0.81 ± 0.05 | 0.56 ± 0.15 | 0.688 |
| Masked MLP, squared error | 0.97 ± 0.01 | 0.91 ± 0.03 | 0.66 ± 0.05 | 0.70 ± 0.01 | 0.92 ± 0.10 | 0.89 ± 0.02 | 0.92 ± 0.06 | 0.96 ± 0.03 | 0.867 |
| Context model k=0 | 0.56 ± 0.03 | 0.72 ± 0.04 | 0.56 ± 0.05 | 0.57 ± 0.02 | 1.00 ± 0.00 | 0.86 ± 0.04 | 0.82 ± 0.04 | 0.52 ± 0.21 | 0.700 |
| Context model k=32 | 0.61 ± 0.08 | 0.73 ± 0.04 | 0.56 ± 0.05 | 0.57 ± 0.02 | 1.00 ± 0.00 | 0.86 ± 0.04 | 0.82 ± 0.03 | 0.61 ± 0.16 | 0.718 |
| Context model k=32, squared error | 0.97 ± 0.01 | 0.93 ± 0.02 | 0.66 ± 0.04 | 0.71 ± 0.01 | 1.00 ± 0.00 | 0.91 ± 0.01 | 0.92 ± 0.03 | 0.98 ± 0.01 | 0.885 |
| kNN to context | 0.98 ± 0.00 | 0.92 ± 0.02 | 0.71 ± 0.02 | 0.75 ± 0.01 | 0.99 ± 0.00 | 0.96 ± 0.01 | 0.84 ± 0.06 | 0.99 ± 0.01 | 0.892 |
| Mahalanobis to context | 0.97 ± 0.01 | 0.92 ± 0.01 | 0.66 ± 0.02 | 0.64 ± 0.01 | 0.98 ± 0.00 | 0.95 ± 0.01 | 0.82 ± 0.06 | 0.97 ± 0.02 | 0.864 |

### ADBench, AUPRC

| method | breastw | cardio | pima | satellite | shuttle | thyroid | vowels | wbc | mean |
|---|---|---|---|---|---|---|---|---|---|
| Isolation Forest | 0.98 ± 0.01 | 0.58 ± 0.07 | 0.51 ± 0.06 | 0.66 ± 0.00 | 0.98 ± 0.01 | 0.66 ± 0.10 | 0.19 ± 0.03 | 0.92 ± 0.11 | 0.684 |
| LOF | 0.32 ± 0.02 | 0.20 ± 0.04 | 0.44 ± 0.02 | 0.38 ± 0.02 | 0.12 ± 0.01 | 0.04 ± 0.01 | 0.38 ± 0.06 | 0.32 ± 0.09 | 0.276 |
| kNN | 0.93 ± 0.03 | 0.32 ± 0.05 | 0.54 ± 0.05 | 0.51 ± 0.01 | 0.18 ± 0.01 | 0.28 ± 0.03 | 0.57 ± 0.12 | 0.86 ± 0.13 | 0.523 |
| PCA | 0.86 ± 0.03 | 0.34 ± 0.03 | 0.44 ± 0.03 | 0.51 ± 0.01 | 0.19 ± 0.02 | 0.56 ± 0.08 | 0.24 ± 0.10 | 0.58 ± 0.19 | 0.465 |
| Masked MLP, standardized error | 0.37 ± 0.02 | 0.54 ± 0.06 | 0.44 ± 0.04 | 0.46 ± 0.02 | 0.66 ± 0.40 | 0.11 ± 0.03 | 0.19 ± 0.02 | 0.10 ± 0.09 | 0.358 |
| Masked MLP, squared error | 0.92 ± 0.02 | 0.56 ± 0.10 | 0.49 ± 0.06 | 0.57 ± 0.01 | 0.71 ± 0.29 | 0.44 ± 0.07 | 0.47 ± 0.17 | 0.63 ± 0.16 | 0.598 |
| Context model k=0 | 0.38 ± 0.07 | 0.51 ± 0.06 | 0.42 ± 0.05 | 0.48 ± 0.02 | 0.95 ± 0.01 | 0.23 ± 0.05 | 0.31 ± 0.03 | 0.21 ± 0.23 | 0.439 |
| Context model k=32 | 0.42 ± 0.13 | 0.53 ± 0.06 | 0.42 ± 0.05 | 0.49 ± 0.02 | 0.96 ± 0.01 | 0.23 ± 0.05 | 0.29 ± 0.04 | 0.22 ± 0.23 | 0.443 |
| Context model k=32, squared error | 0.92 ± 0.03 | 0.59 ± 0.08 | 0.49 ± 0.06 | 0.60 ± 0.01 | 0.93 ± 0.01 | 0.46 ± 0.07 | 0.41 ± 0.09 | 0.74 ± 0.09 | 0.641 |
| kNN to context | 0.96 ± 0.01 | 0.60 ± 0.09 | 0.52 ± 0.04 | 0.67 ± 0.01 | 0.91 ± 0.01 | 0.44 ± 0.08 | 0.20 ± 0.07 | 0.88 ± 0.08 | 0.647 |
| Mahalanobis to context | 0.94 ± 0.02 | 0.54 ± 0.09 | 0.50 ± 0.05 | 0.55 ± 0.01 | 0.78 ± 0.01 | 0.48 ± 0.10 | 0.22 ± 0.09 | 0.71 ± 0.16 | 0.588 |

### AUROC against context size

| k | Context model | kNN to context | Mahalanobis to context | ADBench mean, context model |
|---|---|---|---|---|
| 0 | 0.691 ± 0.022 | n/a | n/a | 0.700 |
| 4 | 0.926 ± 0.020 | 0.916 ± 0.015 | 0.896 ± 0.019 | 0.719 |
| 8 | 0.951 ± 0.016 | 0.949 ± 0.014 | 0.943 ± 0.019 | 0.718 |
| 16 | 0.965 ± 0.015 | 0.966 ± 0.013 | 0.967 ± 0.015 | 0.719 |
| 32 | 0.973 ± 0.011 | 0.978 ± 0.011 | 0.981 ± 0.011 | 0.718 |
| 64 | 0.979 ± 0.005 | 0.986 ± 0.007 | 0.987 ± 0.008 | 0.718 |

### Schema change, half of the columns

| method | synthetic, all columns | synthetic, half | ADBench, all | ADBench, half |
|---|---|---|---|---|
| Isolation Forest | 0.692 ± 0.021 | 0.671 ± 0.012 | 0.877 | 0.845 |
| Local Outlier Factor | 0.691 ± 0.020 | 0.667 ± 0.012 | 0.661 | 0.655 |
| kNN distance | 0.694 ± 0.021 | 0.666 ± 0.013 | 0.828 | 0.802 |
| PCA reconstruction error | 0.583 ± 0.039 | 0.596 ± 0.022 | 0.798 | 0.688 |
| Masked MLP, standardized error | 0.692 ± 0.027 | 0.671 ± 0.015 | 0.688 | 0.736 |
| Masked MLP, squared error | 0.695 ± 0.021 | 0.671 ± 0.014 | 0.867 | 0.831 |
| Context model without context (k=0) | 0.691 ± 0.022 | 0.670 ± 0.014 | 0.700 | 0.717 |
| Context model (k=32) | 0.973 ± 0.011 | 0.925 ± 0.011 | 0.718 | 0.730 |
| Context model, squared error (k=32) | 0.975 ± 0.009 | 0.926 ± 0.010 | 0.885 | 0.840 |
| kNN to the context rows (k=32) | 0.978 ± 0.011 | 0.942 ± 0.014 | 0.892 | 0.856 |
| Mahalanobis to the context rows (k=32) | 0.981 ± 0.011 | 0.935 ± 0.016 | 0.864 | 0.822 |
| Oracle density of the true population | 1.000 ± 0.000 | 0.980 ± 0.005 | n/a | n/a |

### Training cost

| run | context model s | masked MLP s | parameters (context, MLP) |
|---|---|---|---|
| synthetic, one seed | 147 | 99 | 146,452, 142,100 |
| ADBench, both models, one table | 121 |  |  |
