import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib



RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

MODEL_VERSION = "v0.1-synth-logreg"

@dataclass
class DecisionPolicy:
    approve_score: int = 80       # >= approve_score -> approve
    manual_score: int = 55        # >= manual_score -> manual_review, иначе decline

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))

def pd_to_score(pd_prob: np.ndarray) -> np.ndarray:
    """
    Чем ниже PD, тем выше score.
    Простая шкала 0..100: score = round((1 - PD) * 100)
    """
    score = np.round((1.0 - pd_prob) * 100).astype(int)
    return np.clip(score, 0, 100)

def decide(score: int, policy: DecisionPolicy) -> str:
    if score >= policy.approve_score:
        return "approve"
    if score >= policy.manual_score:
        return "manual_review"
    return "decline"

def generate_synthetic_bnpl(n: int = 50_000) -> pd.DataFrame:
    # Базовые признаки
    age = np.random.randint(18, 66, size=n)
    income = np.random.lognormal(mean=np.log(80_000), sigma=0.5, size=n)  # руб/мес
    income = np.clip(income, 25_000, 500_000)

    employment_type = np.random.choice(
        ["full_time", "part_time", "self_employed", "unemployed"],
        size=n,
        p=[0.62, 0.12, 0.20, 0.06]
    )

    region_risk = np.random.choice(["low", "mid", "high"], size=n, p=[0.55, 0.35, 0.10])
    avg_check = np.random.lognormal(mean=np.log(6_000), sigma=0.6, size=n)
    avg_check = np.clip(avg_check, 500, 120_000)

    existing_loans = np.random.poisson(lam=1.2, size=n)
    existing_loans = np.clip(existing_loans, 0, 8)

    overdue_30d = np.random.poisson(lam=0.25, size=n)  # кол-во просрочек 30+ за период
    overdue_30d = np.clip(overdue_30d, 0, 6)

    # Производные признаки
    dti = (existing_loans + 1) * avg_check / income  # грубая "нагрузка"
    dti = np.clip(dti, 0, 2.5)

    # --- "Истинная" вероятность дефолта (синтетическая логика) ---
    # Это имитирует бизнес-реальность: просрочки, низкий доход, безработица, высокий DTI увеличивают риск
    emp_map = {"full_time": -0.6, "part_time": -0.2, "self_employed": 0.1, "unemployed": 0.9}
    reg_map = {"low": -0.3, "mid": 0.0, "high": 0.35}

    logit = (
        -2.2
        + 1.2 * overdue_30d
        + 1.8 * dti
        + 0.35 * existing_loans
        + np.vectorize(emp_map.get)(employment_type)
        + np.vectorize(reg_map.get)(region_risk)
        + 0.015 * (30 - np.minimum(age, 30))  # чуть выше риск у самых молодых
        - 0.000008 * (income - 80_000)        # чем выше доход — тем ниже риск
    )

    pd_true = sigmoid(logit)
    # Генерируем бинарную метку дефолта из вероятности
    default = (np.random.rand(n) < pd_true).astype(int)

    df = pd.DataFrame({
        "age": age,
        "income": income.astype(int),
        "employment_type": employment_type,
        "region_risk": region_risk,
        "avg_check": avg_check.astype(int),
        "existing_loans": existing_loans,
        "overdue_30d": overdue_30d,
        "dti": dti,
        "default": default,
    })
    return df

def main():
    df = generate_synthetic_bnpl(n=50_000)

    # One-hot для категорий
    X = df.drop(columns=["default"])
    y = df["default"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    num_cols = ["age", "income", "avg_check", "existing_loans", "overdue_30d", "dti"]
    cat_cols = ["employment_type", "region_risk"]

    preprocess = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
        ]
    )

    model = Pipeline(steps=[
        ("preprocess", preprocess),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    model.fit(X_train, y_train)

    pd_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, pd_pred)
    print(f"ROC-AUC: {auc:.4f}")

    # Переводим в score и решения
    scores = pd_to_score(pd_pred)
    policy = DecisionPolicy(approve_score=80, manual_score=55)
    decisions = np.array([decide(s, policy) for s in scores])

    eval_df = pd.DataFrame({
        "y_true": y_test,
        "pd": pd_pred,
        "score": scores,
        "decision": decisions,
    })

    summary = eval_df.groupby("decision").agg(
        cnt=("y_true", "size"),
        default_rate=("y_true", "mean"),
        avg_pd=("pd", "mean"),
        min_score=("score", "min"),
        max_score=("score", "max"),
    ).sort_values("default_rate")

    def risk_class_from_score(score: int) -> str:
        if score >= 80:
            return "low"
        if score >= 55:
            return "medium"
        return "high"

    def predict_payload(model, client_df: pd.DataFrame, policy: DecisionPolicy) -> dict:
        pd_prob = float(model.predict_proba(client_df)[:, 1][0])
        score = int(np.clip(round((1 - pd_prob) * 100), 0, 100))
        decision = decide(score, policy)
        risk_class = risk_class_from_score(score)
        return {
            "pd": pd_prob,
            "score": score,
            "risk_class": risk_class,
            "decision": decision,
            "model_version": MODEL_VERSION,
        }

    print("\nBucket quality:")
    print(summary)

    # Быстрый отчёт по распределению решений
    unique, counts = np.unique(decisions, return_counts=True)
    print("Decision distribution:", dict(zip(unique, counts)))

    # Классификационный отчёт на пороге (для ориентира)
    # Превратим score обратно в "предсказание дефолта" на условном пороге:
    y_hat = (pd_pred >= 0.5).astype(int)
    print(classification_report(y_test, y_hat, digits=4))

    # Пример 5 строк результата
    sample = X_test.head(5).copy()
    sample["pd"] = pd_pred[:5]
    sample["score"] = scores[:5]
    sample["decision"] = decisions[:5]
    print(sample[["pd", "score", "decision"]])

    # --- Save trained pipeline ---
    joblib.dump(model, "scoring_model.joblib")
    print("\nSaved model: scoring_model.joblib")

    # --- Load and predict one sample (mini inference test) ---
    loaded = joblib.load("scoring_model.joblib")

    one_client = pd.DataFrame([{
        "age": 28,
        "income": 65000,
        "employment_type": "full_time",
        "region_risk": "mid",
        "avg_check": 8500,
        "existing_loans": 2,
        "overdue_30d": 0,
        "dti": 0.4,
    }])

    pd_one = loaded.predict_proba(one_client)[:, 1][0]
    score_one = int(np.clip(round((1 - pd_one) * 100), 0, 100))

    policy = DecisionPolicy(approve_score=80, manual_score=55)
    payload = predict_payload(loaded, one_client, policy)
    print("\nOne client inference:")
    print(payload)

    test_clients = pd.DataFrame([
    # низкий риск
    {"age": 35, "income": 140000, "employment_type": "full_time", "region_risk": "low",
     "avg_check": 6000, "existing_loans": 1, "overdue_30d": 0, "dti": 0.15},

    # средний риск
    {"age": 28, "income": 65000, "employment_type": "full_time", "region_risk": "mid",
     "avg_check": 8500, "existing_loans": 2, "overdue_30d": 0, "dti": 0.4},

    # высокий риск
    {"age": 22, "income": 45000, "employment_type": "unemployed", "region_risk": "high",
     "avg_check": 18000, "existing_loans": 4, "overdue_30d": 2, "dti": 1.4},
    ])

    print("\n3 clients inference:")
    for i in range(len(test_clients)):
        row = test_clients.iloc[[i]]
        print(i + 1, predict_payload(loaded, row, policy))



if __name__ == "__main__":
    main()
