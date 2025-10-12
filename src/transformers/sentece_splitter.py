import re

concatenated_pattern = re.compile(r"(\w+[^\s\w]+\w{3,}[^\s\w]*)+(?!\s)")
separators_pattern = re.compile(r'[^\w\s]+')

def fix_concatenated_words(X):
    result = []
    for text in X:
        matches = list(concatenated_pattern.finditer(text))
        for m in reversed(matches):
            problematic_sub = m.group(0)
            separators = separators_pattern.findall(problematic_sub)
            separators = list(set(separators))
            separators.sort(reverse=True, key=len)
            escaped_seps = '|'.join(re.escape(s) for s in separators)
            fixed_sub = re.sub(f'({escaped_seps})', r'\1 ', problematic_sub)
            text = text.replace(problematic_sub, fixed_sub)
        result.append(text.strip())
    return result
