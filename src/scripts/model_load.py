# pylint: disable=unused-import

import joblib
from sklearn.preprocessing import LabelEncoder
from src.util.path_helper import PathHelper
from src.pipelines import (
    preprocessing_pieline,
    text_vecrotization_pipeline,
    classification_pipeline
)

label_encoder = joblib.load(PathHelper.models.label_encoder)
preprocessor = joblib.load(PathHelper.models.base_text_preprocessor)
vectorizer = joblib.load(PathHelper.models.vectorizer)
classifier = joblib.load(PathHelper.models.sbert_classifier)

def predict(X):
    preprocessed = preprocessor.transform(X)
    vectorized = vectorizer.transform(preprocessed)
    predicted = classifier.predict(vectorized)
    decoded = label_encoder.inverse_transform(predicted)
    return decoded
