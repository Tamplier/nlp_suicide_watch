from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from xgboost import XGBClassifier
from src.transformers import (
    fix_concatenated_words, SpacyTokenizer, ExtraFeatures,
    FeatureSelector, SbertVectorizer, fix_feature_names
)

def preprocessing_pieline(top_k_feat=15):
    extra_features_routine = Pipeline([
        ('selector', FeatureSelector(top_k_feat)),
        ('scaler', StandardScaler().set_output(transform="pandas")),
    ])
    col_transformer = ColumnTransformer([
        ('extra_features_routine', extra_features_routine, selector(dtype_include='number'))
    ], remainder='passthrough')
    extra_features_routine.set_output(transform='pandas')
    col_transformer.set_output(transform='pandas')
    return Pipeline([
        ('splitter', FunctionTransformer(fix_concatenated_words, validate=False)),
        ('tokenizer', SpacyTokenizer()),
        ('features_extractor', ExtraFeatures()),
        ('column_transformer', col_transformer)
    ])

def text_vecrotization_pipeline():
    vectorizer = ColumnTransformer([
        ('sbert_vectorize', SbertVectorizer(), 'text')
    ], remainder='passthrough')
    return Pipeline([
        ('fix_column_names', FunctionTransformer(fix_feature_names, validate=False)),
        ('vectorize', vectorizer)
    ])

def classification_pipeline(params):
    return Pipeline([
        ("clf", XGBClassifier(**params))
    ])
