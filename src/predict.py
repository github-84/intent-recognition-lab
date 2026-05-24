from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "intent_model.joblib"


INTENT_NAMES = {
    "query_order": "查询订单",
    "cancel_order": "取消订单",
    "refund": "退款售后",
    "product_consult": "商品咨询",
    "human_service": "人工客服",
    "greeting": "问候闲聊",
}


@lru_cache(maxsize=1)
def load_model(model_path: str = str(MODEL_PATH)) -> dict[str, Any]:
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model file not found: {path}. Run `python src/train.py` first."
        )
    bundle = joblib.load(path)
    if "pipeline" not in bundle:
        raise ValueError("Invalid model bundle: missing pipeline.")
    return bundle


def predict_intent(text: str) -> dict[str, Any]:
    cleaned_text = text.strip()
    if not cleaned_text:
        raise ValueError("Input text cannot be empty.")

    bundle = load_model()
    pipeline = bundle["pipeline"]
    labels = list(pipeline.classes_)
    probabilities = pipeline.predict_proba([cleaned_text])[0]
    ranked = sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True)
    intent, confidence = ranked[0]

    return {
        "text": cleaned_text,
        "intent": intent,
        "intent_name": INTENT_NAMES.get(intent, intent),
        "confidence": round(float(confidence), 4),
        "probabilities": {label: round(float(score), 4) for label, score in ranked},
        "probability_names": {
            label: INTENT_NAMES.get(label, label) for label, _score in ranked
        },
    }


if __name__ == "__main__":
    while True:
        try:
            user_text = input("请输入用户话术，输入 q 退出：").strip()
        except EOFError:
            break
        if user_text.lower() in {"q", "quit", "exit"}:
            break
        try:
            result = predict_intent(user_text)
            print(result)
        except Exception as exc:
            print(f"预测失败：{exc}")
