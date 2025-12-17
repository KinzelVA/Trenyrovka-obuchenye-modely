import os
import json
import time
import uuid
import hashlib
from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
import joblib

from fastapi import FastAPI, Header, Request, HTTPException
from pydantic import BaseModel

from sqlalchemy import create_engine, text


# ----------------------------
# Model config
# ----------------------------
MODEL_VERSION = "v0.1-synth-logreg"
MODEL_PATH = "scoring_model.joblib"


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


MODEL_HASH = _sha256_file(MODEL_PATH)
# детерминированный UUID модели (один и тот же при одном и том же файле)
MODEL_ID = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{MODEL_VERSION}:{MODEL_HASH}"))


def decide(score: int, approve_score: int = 80, manual_score: int = 55) -> str:
    if score >= approve_score:
        return "approve"
    if score >= manual_score:
        return "manual_review"
    return "decline"


def risk_class_from_score(score: int) -> str:
    if score >= 80:
        return "low"
    if score >= 55:
        return "medium"
    return "high"


def pd_to_score(pd_prob: float) -> int:
    return int(np.clip(round((1 - pd_prob) * 100), 0, 100))


class ScoreRequest(BaseModel):
    age: int
    income: int
    employment_type: str
    region_risk: str
    avg_check: int
    existing_loans: int
    overdue_30d: int
    dti: float


app = FastAPI(title="BNPL Scoring API", version=MODEL_VERSION)

model = joblib.load(MODEL_PATH)


# ----------------------------
# DB / Audit
# ----------------------------
DATABASE_URL = os.getenv("DATABASE_URL")  # пример: postgresql+psycopg2://user:pass@127.0.0.1:5433/scoring
_engine = None


def _get_engine():
    global _engine
    if _engine is not None:
        return _engine
    if not DATABASE_URL:
        return None
    _engine = create_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


def _db_configured() -> bool:
    return _get_engine() is not None


def _init_db():
    eng = _get_engine()
    if eng is None:
        return

    ddl = """
    CREATE TABLE IF NOT EXISTS public.scoring_audit_log (
        id BIGSERIAL PRIMARY KEY,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),

        request_id UUID,
        model_version TEXT,
        model_id UUID,

        decision TEXT,
        score INT,
        pd DOUBLE PRECISION,
        risk_class TEXT,

        request_json JSONB,
        response_json JSONB,

        client_id TEXT,
        source TEXT,
        latency_ms INT,
        error_reason TEXT,

        idempotency_key TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_audit_created_at ON public.scoring_audit_log (created_at);
    CREATE INDEX IF NOT EXISTS idx_audit_decision ON public.scoring_audit_log (decision);
    CREATE INDEX IF NOT EXISTS idx_audit_idem_key ON public.scoring_audit_log (idempotency_key);
    """
    with eng.begin() as conn:
        conn.execute(text(ddl))


def _audit_insert(payload: Dict[str, Any]) -> None:
    eng = _get_engine()
    if eng is None:
        return

    sql = text("""
        INSERT INTO public.scoring_audit_log (
            request_id, model_version, model_id,
            decision, score, pd, risk_class,
            request_json, response_json,
            client_id, source, latency_ms, error_reason,
            idempotency_key
        )
        VALUES (
            :request_id, :model_version, :model_id,
            :decision, :score, :pd, :risk_class,
            CAST(:request_json AS JSONB), CAST(:response_json AS JSONB),
            :client_id, :source, :latency_ms, :error_reason,
            :idempotency_key
        )
    """)
    with eng.begin() as conn:
        conn.execute(sql, payload)


def _maybe_idempotent_replay(idempotency_key: Optional[str], source: Optional[str] = None):
    if not idempotency_key:
        return None
    eng = _get_engine()
    if eng is None:
        return None

    if source:
        sql = text("""
            SELECT response_json
            FROM public.scoring_audit_log
            WHERE idempotency_key = :k AND source = :s AND error_reason IS NULL
            ORDER BY id DESC
            LIMIT 1
        """)
        params = {"k": idempotency_key, "s": source}
    else:
        sql = text("""
            SELECT response_json
            FROM public.scoring_audit_log
            WHERE idempotency_key = :k AND error_reason IS NULL
            ORDER BY id DESC
            LIMIT 1
        """)
        params = {"k": idempotency_key}

    with eng.begin() as conn:
        row = conn.execute(sql, params).fetchone()

    if not row:
        return None

    # response_json может вернуться как dict (jsonb) или как строка — нормализуем
    val = row[0]
    if isinstance(val, (dict, list)):
        return val
    try:
        return json.loads(val)
    except Exception:
        return val


def _score_one(
    req_dict: Dict[str, Any],
    *,
    idempotency_key: Optional[str],
    client_id: Optional[str],
    source: Optional[str],
) -> Dict[str, Any]:
    t0 = time.perf_counter()

    # идемпотентность (один клиентский запрос — один результат)
    replay = _maybe_idempotent_replay(idempotency_key)
    if replay is not None:
        return replay

    request_id = str(uuid.uuid4())

    try:
        df = pd.DataFrame([req_dict])
        pd_prob = float(model.predict_proba(df)[:, 1][0])
        score_val = pd_to_score(pd_prob)
        decision = decide(score_val)
        risk_class = risk_class_from_score(score_val)

        response = {
            "request_id": request_id,
            "pd": pd_prob,
            "score": score_val,
            "risk_class": risk_class,
            "decision": decision,
            "model_version": MODEL_VERSION,
            "model_id": MODEL_ID,
        }

        latency_ms = int((time.perf_counter() - t0) * 1000)

        _audit_insert({
            "request_id": request_id,
            "model_version": MODEL_VERSION,
            "model_id": MODEL_ID,
            "decision": decision,
            "score": score_val,
            "pd": pd_prob,
            "risk_class": risk_class,
            "request_json": json.dumps(req_dict, ensure_ascii=False),
            "response_json": json.dumps(response, ensure_ascii=False),
            "client_id": client_id,
            "source": source,
            "latency_ms": latency_ms,
            "error_reason": None,
            "idempotency_key": idempotency_key,
        })

        return response

    except Exception as e:
        latency_ms = int((time.perf_counter() - t0) * 1000)
        _audit_insert({
            "request_id": request_id,
            "model_version": MODEL_VERSION,
            "model_id": MODEL_ID,
            "decision": "error",
            "score": -1,
            "pd": None,
            "risk_class": None,
            "request_json": json.dumps(req_dict, ensure_ascii=False),
            "response_json": json.dumps({"error": str(e)}, ensure_ascii=False),
            "client_id": client_id,
            "source": source,
            "latency_ms": latency_ms,
            "error_reason": str(e),
            "idempotency_key": idempotency_key,
        })
        raise


@app.on_event("startup")
def on_startup():
    if _db_configured():
        _init_db()


# ----------------------------
# Endpoints
# ----------------------------
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_version": MODEL_VERSION,
        "model_id": MODEL_ID,
        "db_configured": _db_configured(),
        "audit_enabled": _db_configured(),
    }


@app.post("/score")
def score(
    req: ScoreRequest,
    request: Request,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
    source: Optional[str] = Header(default=None, alias="X-Source"),
):
    # именно эти Header-параметры заставляют Swagger показать поля заголовков
    req_dict = req.model_dump()
    try:
        return _score_one(
            req_dict,
            idempotency_key=idempotency_key,
            client_id=client_id,
            source=source,
        )
    except Exception:
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/score/batch")
def score_batch(
    reqs: List[ScoreRequest],
    request: Request,
    idempotency_key: Optional[str] = Header(default=None, alias="Idempotency-Key"),
    client_id: Optional[str] = Header(default=None, alias="X-Client-Id"),
    source: Optional[str] = Header(default=None, alias="X-Source"),
):
    # идемпотентность для всего батча: если батч с таким ключом уже был — вернем сохраненный ответ
    replay = _maybe_idempotent_replay(idempotency_key, source="batch")
    if replay is not None:
        return replay

    try:
        items = []
        for i, r in enumerate(reqs, start=1):
            item_key = f"{idempotency_key}:{i}" if idempotency_key else None
            items.append(
                _score_one(
                    r.model_dump(),
                    idempotency_key=item_key,
                    client_id=client_id,
                    source=source,
                )
            )

        resp = {
            "batch_id": str(uuid.uuid4()),
            "count": len(items),
            "items": items,
            "model_version": MODEL_VERSION,
            "model_id": MODEL_ID,
        }

        # сохраняем "сводную" запись батча (чтобы replay работал по одному ключу)
        _audit_insert({
            "request_id": str(uuid.uuid4()),
            "model_version": MODEL_VERSION,
            "model_id": MODEL_ID,
            "decision": "batch",
            "score": -1,
            "pd": None,
            "risk_class": None,
            "request_json": json.dumps([r.model_dump() for r in reqs], ensure_ascii=False),
            "response_json": json.dumps(resp, ensure_ascii=False),
            "client_id": client_id,
            "source": "batch",
            "latency_ms": None,
            "error_reason": None,
            "idempotency_key": idempotency_key,
        })

        return resp

    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal Server Error")
