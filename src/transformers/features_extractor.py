import logging
import re
import gc
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from emot import emot
from emot.emo_unicode import EMOTICONS_EMO
from urlextract import URLExtract
from src.util.typos_processor import typos_processor

logger = logging.getLogger(__name__)

class ExtraFeatures(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.repeat_pattern = re.compile(r'(\w)\1{2,}', re.IGNORECASE)
        self.censorshop_pattern = re.compile(r"[^\w\s'\-:]+")
        self.emot_obj = emot()
        self.emotions = set(EMOTICONS_EMO.keys())
        self.emot_meanings = set(EMOTICONS_EMO.values())
        self.url_extractor = URLExtract()
        self.feature_names_ = [
            "text", "length", "upcase_rate", "exc_mark_rate", "q_mark_rate", "dots_rate",
            "new_lines_rate", "median_sentence_len", "sentences_count", "urls_counter",
            "censured", "compression"
        ] + sorted(self.emot_meanings)

    def get_feature_names_out(self, input_features=None):
        return self.feature_names_

    def fit(self, X, y=None):
        return self

    def replace_urls(self, text):
        urls = self.url_extractor.find_urls(text)
        urls = [u.rstrip('.,!?:;') for u in urls]
        urls_counter = len(urls)
        result = text
        for url in urls:
            result = result.replace(url, '[l]')
        return result, urls_counter

    def sentences_stat(self, doc):
        sentences_len = np.array([len(sent.text.strip()) for sent in doc.sents])
        sentences_len = sentences_len[sentences_len > 0]
        return {
            'median_sentence_len': np.nan_to_num(np.median(sentences_len)),
            'sentences_count': len(sentences_len)
        }

    def base_stat(self, text, sentences_count):
        sentences_count = max(1, sentences_count)
        text_len = max(1, len(text))
        exc_marks = text.count('!')
        q_marks = text.count('?')
        dots = text.count('.')
        new_lines = text.count('\n')
        upcase = sum(1 for c in text if c.isupper())

        return {
            'length': text_len,
            'upcase_rate': upcase / sentences_count,
            'exc_mark_rate': exc_marks / sentences_count,
            'q_mark_rate': q_marks / sentences_count,
            'dots_rate': dots / sentences_count,
            'new_lines_rate': new_lines / sentences_count
        }

    def emoticons_stat(self, text):
        emot_counts = Counter({item: 0 for item in self.emot_meanings})
        detected = self.emot_obj.emoticons(text)
        if detected['flag']:
            emot_counts.update(detected['mean'])
        return emot_counts

    def typos_stat_and_fix(self, doc):
        tokens = [token for token in doc if not token.is_punct]

        emot_cache = {}
        def get_emoticon_flag(text):
            if text not in emot_cache:
                emot_cache[text] = self.emot_obj.emoticons(text)['flag']
            return emot_cache[text]

        tokens = [t.lemma_.lower() if not get_emoticon_flag(t.text) else t.text for t in tokens]
        censorship = sum(1 for t in tokens if not get_emoticon_flag(t) and self.censorshop_pattern.findall(t))
        corrected_text = typos_processor(doc)
        return {
            'text': corrected_text,
            'censured': censorship,
            'compression': max(0, len(doc.text) - len(corrected_text))
        }

    def transform(self, X):
        feats = []
        logger.info('Extra features extraction start')
        total_messages = len(X)
        milestones = [0.25, 0.5, 0.75]
        real_milestones = [int(total_messages * m) for m in milestones]
        for i, doc in enumerate(X):
            if i in real_milestones:
                j = real_milestones.index(i)
                logger.info('Extra features finalized %f of total records', milestones[j])
            text = doc.text
            text, urls_counter = self.replace_urls(text)
            sentence_feats = self.sentences_stat(doc)
            base_feats = self.base_stat(text, sentence_feats['sentences_count'])
            emot_feats = self.emoticons_stat(text)
            typos_feats = self.typos_stat_and_fix(doc)

            feats_row = {
                **base_feats,
                **sentence_feats,
                "urls_counter": urls_counter,
                **emot_feats,
                **typos_feats
            }
            row = [feats_row.get(name, 0) for name in self.feature_names_]
            feats.append(row)
        gc.collect()
        logger.info('Extra features extraction finish')
        result = pd.DataFrame(feats, columns=self.feature_names_)
        return result
