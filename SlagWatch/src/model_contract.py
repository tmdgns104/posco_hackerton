"""Shared model interface and compatibility wrapper for existing image models."""
from abc import ABC, abstractmethod
import time

import numpy as np


class BaseModelAdapter(ABC):
    @abstractmethod
    def validate_model(self):
        raise NotImplementedError

    @abstractmethod
    def get_metadata(self):
        raise NotImplementedError

    @abstractmethod
    def validate_input(self, data):
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, data):
        raise NotImplementedError

    @abstractmethod
    def predict(self, data):
        raise NotImplementedError

    @abstractmethod
    def normalize_output(self, *args):
        raise NotImplementedError

    @abstractmethod
    def infer(self, sample_ids, data):
        raise NotImplementedError


class ImageModelAdapter(BaseModelAdapter):
    def __init__(self, adapter, model_id, classes, kind):
        if kind not in ("binary", "multiclass"):
            raise ValueError("Unsupported image model kind.")
        self.adapter = adapter
        self.model_id = model_id
        self.classes = list(classes)
        self.kind = kind
        self.identity = adapter.identity
        self.validate_model()

    def validate_model(self):
        if not isinstance(self.identity, dict) or not callable(getattr(self.adapter, "predict", None)):
            raise ValueError("Image adapter identity or prediction method is missing.")
        return self.identity

    def get_metadata(self):
        return {"model_id": self.model_id, "model_type": "image_classification",
                "classes": self.classes, "input_schema": "RGB HWC float32" if self.kind == "binary"
                else "200x200 grayscale float32", "identity": self.identity}

    def validate_input(self, data):
        if not isinstance(data, (list, tuple)):
            raise ValueError("Image input must be a sequence.")
        for raw in data:
            shape_ok = (isinstance(raw, np.ndarray) and
                        ((self.kind == "binary" and raw.ndim == 3 and raw.shape[2] == 3 and min(raw.shape[:2]) > 0)
                         or (self.kind == "multiclass" and raw.shape == (200, 200))))
            if (not shape_ok or raw.dtype != np.float32 or not np.isfinite(raw).all()
                    or raw.min() < 0 or raw.max() > 1):
                raise ValueError("Image pixels do not match the model input contract.")
        return data

    def preprocess(self, data):
        return self.validate_input(data)

    def normalize_output(self, output):
        return output

    def predict(self, data):
        return self.normalize_output(self.adapter.predict(self.preprocess(data)))

    def infer(self, sample_ids, data):
        if len(sample_ids) != len(data):
            raise ValueError("Sample IDs and input count differ.")
        rows = []
        for sample_id, raw in zip(sample_ids, data, strict=True):
            started = time.perf_counter()
            result = self.predict([raw])
            if len(result) != 1:
                raise ValueError("Image adapter must return one prediction per input.")
            item = result[0]
            rows.append({"sample_id": sample_id, "model_id": self.model_id,
                         "predicted_class": item["predicted_label"], "scores": item["scores"],
                         "score_type": item["score_type"], "inference_status": item["status"],
                         "error_code": "", "inference_duration_ms": (time.perf_counter()-started)*1000})
        return rows
