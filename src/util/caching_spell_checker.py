from functools import lru_cache
from pathlib import Path
from symspellpy.symspellpy import SymSpell, Verbosity
import hunspell

class CachingSpellChecker:
    def __init__(self):
        self.hunspell = hunspell.HunSpell(
            '/usr/share/hunspell/en_US.dic',
            '/usr/share/hunspell/en_US.aff'
        )
        self.prepare_symspell()

    def prepare_symspell(self):
        base_dir = Path(__file__).resolve().parent.parent.parent
        dict_path = base_dir / 'resources/frequency_dictionary_en_82_765.txt'
        self.symspell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        self.symspell.load_dictionary(dict_path, term_index=0, count_index=1)

    def correct_words(self, words):
        unique_words = set(words)
        corrections = [self._correct_word(w) for w in unique_words]
        corrections = dict(corrections)
        return [corrections.get(w, w) for w in words]

    @lru_cache(maxsize=10000)
    def _correct_word(self, word):
        is_correct = self.hunspell.spell(word)
        corrected = word
        if not is_correct:
            ss_suggestions = self.symspell.lookup(word, Verbosity.CLOSEST, max_edit_distance=2)
            hs_suggestions = self.hunspell.suggest(word)
            corrected = (
                ss_suggestions[0].term
                if ss_suggestions
                else hs_suggestions[0]
                if hs_suggestions
                else word
            )
        return word, corrected
