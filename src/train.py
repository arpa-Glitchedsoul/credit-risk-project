from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier


def make_logreg():
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # learned from training folds only
        ("scale", StandardScaler()),                    # puts features on a similar scale
        ("model", LogisticRegression(class_weight="balanced",
                                     max_iter=1000, random_state=42)),
    ])


def make_logreg_smote():
    return ImbPipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("smote", SMOTE(random_state=42)),     # runs on training folds only
        ("model", LogisticRegression(max_iter=1000, random_state=42)),
    ])
def make_lgbm(**params):
    return LGBMClassifier(class_weight="balanced", random_state=42,
                          n_jobs=1, verbose=-1, **params)