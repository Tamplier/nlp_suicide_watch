import pytest
import spacy
import numpy as np
from src.transformers.features_extractor import ExtraFeatures
from src.transformers.spacy_tokenizer import SpacyTokenizer

extractor = ExtraFeatures()
nlp = spacy.load('en_core_web_sm', disable=["ner", "textcat"])

@pytest.mark.parametrize(
    'text,expected',
    [
        ('Image: http://www.test.com/img?id=5. Upvote it!', ('Image: [l]. Upvote it!', 1)),
        ('Profile: test.com/u/123. Profile: 127.0.0.1/u?id=1', ('Profile: [l]. Profile: [l]', 2)),
        ("Don't do that", ("Don't do that", 0))
    ]
)
def test_replace_urls(text, expected):
    result = extractor.replace_urls(text)
    assert result == expected

@pytest.mark.parametrize(
    'text,median_len,sentences_count',
    [
        ('Hello!! My name is Jonas.', 12, 2),
        ('', 0, 0)
    ]
)
def test_sentences_stat(text, median_len, sentences_count):
    doc = nlp(text)
    result = extractor.sentences_stat(doc)
    assert result['median_sentence_len'] == median_len
    assert result['sentences_count'] == sentences_count

@pytest.mark.parametrize(
    'text,sentences,expected',
    [
        (
            'Hello\n This is a typical neutral message.',
            2,
            {
                'length': 41,
                'upcase_rate': 1,
                'exc_mark_rate': 0,
                'q_mark_rate': 0,
                'dots_rate': 0.5,
                'new_lines_rate': 0.5
            }
        ),
        (
            "HI!!! I'M SOOOO EXCITED!!!! WHAT ARE YOU DOING?????",
            3,
            {
                'length': 51,
                'upcase_rate': 10.33,
                'exc_mark_rate': 2.33,
                'q_mark_rate': 1.66,
                'dots_rate': 0,
                'new_lines_rate': 0
            }
        )
    ]
)
def test_base_stat(text, sentences, expected):
    result = extractor.base_stat(text, sentences)
    for key in expected:
        assert result[key] == pytest.approx(expected[key], abs=0.01)

@pytest.mark.parametrize(
    'text,emotion,count',
    [
        ("Hey, how you're doing ;)", 'Wink or smirk', 1),
        ("It's so stuped lol :-)))", 'Very very Happy face or smiley', 1),
        ('Love you :* :* :*', 'Kiss', 3)
    ]
)
def test_emoticons_stat(text, emotion, count):
    result = extractor.emoticons_stat(text)
    assert result[emotion] == count

@pytest.mark.parametrize(
    'text, expected',
    [
        (
            'Stop f*ck my brain!!!!!',
            {
                'text': 'stop fuck my brain!!',
                'censured': 1,
                'compression': 3
            }
        ),
        (
            "I'm sooooooo exc!ted!",
            {
                'text': "i'm so excited!",
                'censured': 1,
                'compression': 6
            }
        ),
        (
            "Don't do that. He's my co-worker.",
            {
                'text': "don't do that. he's my co-worker.",
                'censured': 0,
                'compression': 0
            }
        ),
    ]
)
def test_typos_stat_and_fix(text, expected):
    doc = nlp(text)
    result = extractor.typos_stat_and_fix(doc)
    assert result == expected

@pytest.mark.parametrize(
    'text,expected',
    [
        (
            "Oh f*ck!!!!! It's really surpr!sing. oO (o.o)",
            {
                'text': "oh fuck!! it's really surprising. oO (o.o)",
                'censured': 2,
                'Surprised': 2,
                'length': 45,
                'sentences_count': 3,
                'exc_mark_rate': 2.0
            }
        )
    ]
)
def test_transform(text, expected):
    tokenizer = SpacyTokenizer()
    docs = tokenizer.transform([text])
    result = extractor.transform(docs).iloc[0]
    features = extractor.get_feature_names_out()
    columns = list(result.index)
    np.testing.assert_array_equal(features, columns)
    for key in expected:
        assert result[key] == expected[key]
