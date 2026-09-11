from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


def train_models(X_train, y_train, preprocessor):
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
        ),
    }
    pipelines = {}
    for name, model in models.items():
        pipeline = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        pipelines[name] = pipeline
    return pipelines
