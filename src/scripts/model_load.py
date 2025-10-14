# pylint: disable=unused-import

import joblib
import torch
from sklearn.preprocessing import LabelEncoder
from src.util import PathHelper, GPUManager
from src.pipelines import (
    preprocessing_pieline,
    text_vecrotization_pipeline,
    classification_pipeline
)

# Hack to load model without GPU
original_torch_load = torch.load
device = GPUManager.device()
torch.load = lambda f, *args, **kwargs: original_torch_load(f, map_location=device, *args, **kwargs)

label_encoder = joblib.load(PathHelper.models.label_encoder)
preprocessor = joblib.load(PathHelper.models.base_text_preprocessor)
vectorizer = joblib.load(PathHelper.models.vectorizer, mmap_mode=None)
classifier = joblib.load(PathHelper.models.sbert_classifier, mmap_mode=None)

def predict(X):
    preprocessed = preprocessor.transform(X)
    vectorized = vectorizer.transform(preprocessed)
    predicted = classifier.predict(vectorized)
    decoded = label_encoder.inverse_transform(predicted)
    return decoded
