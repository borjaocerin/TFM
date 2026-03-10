from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

from app.core.config import settings


class ModelStore:
    def __init__(self, model_dir: Path | None = None) -> None:
        self.model_dir = model_dir or settings.model_dir
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.model_path = self.model_dir / "model.pkl"
        self.metadata_path = self.model_dir / "metadata.json"

    def save(self, model_payload: dict[str, Any], metadata: dict[str, Any]) -> tuple[Path, Path]:
        joblib.dump(model_payload, self.model_path)
        self.metadata_path.write_text(
            json.dumps(metadata, indent=2, ensure_ascii=True),
            encoding="utf-8",
        )
        return self.model_path, self.metadata_path

    def load(self) -> tuple[dict[str, Any], dict[str, Any]]:
        if not self.model_path.exists() or not self.metadata_path.exists():
            raise FileNotFoundError("No existe un modelo activo en el model store")
        payload = joblib.load(self.model_path)
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        return payload, metadata
