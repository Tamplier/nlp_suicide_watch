import re
from emot import emot
from src.util import CachingSpellChecker

repeat_pattern = re.compile(r'(\w|[^\w\d\s])\1{2,}', re.IGNORECASE)
spell_checker = CachingSpellChecker()
emot_obj = emot()

def typos_processor(doc):
    tokens = []
    word_tokens = []
    for token in doc:
        emotion = emot_obj.emoticons(token.text)['flag']
        token_l = token.text
        if not emotion:
            token_l = repeat_pattern.sub(r'\1\1', token.lower_)
            if not token.is_stop and not token.is_punct and token.lemma_.strip():
                word_tokens.append(token_l)
        tokens.append(token_l)
    corrected_words = spell_checker.correct_words(word_tokens)
    mapper = dict(zip(word_tokens, corrected_words))
    tokens = [mapper.get(t, t) for t in tokens]
    tokens = [proc_tok + orig_tok.whitespace_ for proc_tok, orig_tok in zip(tokens, doc)]
    text = ''.join(tokens)
    result = repeat_pattern.sub(r'\1\1', text)
    return result
