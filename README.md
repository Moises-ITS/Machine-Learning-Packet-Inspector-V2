# Machine-Learning-Packet-Inspector-V2
# Network Intrusion Detection (V2) - XGBoost Pipeline

## 🛡️ Project Overview
This project implements a high-performance Intrusion Detection System (IDS) using XGBoost. By moving beyond basic classification, this version utilizes a **ColumnTransformer Pipeline** and **Stratified Shuffle Splitting** to ensure robust detection of modern network threats.

## 🚀 Key Improvements in V2
* **Stratified Sampling:** Merged training/testing sets to perform a custom 80/20 stratified split, ensuring consistent attack-to-normal ratios across sets.
* **XGBoost Optimization:** Implemented Gradient Boosting with a tuned learning rate (0.05) and 500+ estimators.
* **Production Pipeline:** Built a scikit-learn `Pipeline` to automate StandardScaler and OneHotEncoding, preventing data leakage.
* **Class Imbalance Handling:** Utilized `scale_pos_weight` to prioritize the detection of rare, high-risk attack categories.

## 📊 Performance Analysis
I compared a "Fast" learning rate (0.2) vs. a "Conservative" learning rate (0.05):
| Metric | LR 0.2 | LR 0.05 |
| **False Positives** | 621 | 566 |
| **False Negatives** | 1,971 | 2,289 |

*Note: While 0.05 reduced false alarms, 0.2 proved more aggressive in catching stealthy attacks.*

#Notes to Self

- There is never a balance; instead, some sacrifice is required to ensure "near perfect" results, such as  in the "paranoid mode" matrix, where threshold sensitivity was changed from 0.5 to 0.3 and accuracy diminished, but False negative rates decreased
- Using the correct classification model matters more than tweaking settings; changing from RandomForest to XGBoost alone gave me an improved accuracy of up to 4%, and using StratifiedShuffleSplit gave me an additional 2% from my previous model.
- Improving the False Positive Rates and False Negative rates has more to do with tweaking model settings such as n_estimators, learning_rate, max_depth, etc. This is different from accuracy metrics, where models, split,s and encoding make a bigger difference
