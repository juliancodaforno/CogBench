import re

def remove_repeated_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    result = []
    for sentence in sentences:
        if sentence not in seen:
            seen.add(sentence)
            result.append(sentence)
    if len(result) > 1:
        final = ' '.join(result[:-2]) + ' ' + '\n'.join(result[-2:])
    else:
        final = ' '.join(result)
    
    return final

def remove_excessive_newlines(text):
    # Replace any occurrence of three or more newline characters with two newline characters
    text = re.sub('\n{3,}', '\n\n', text)
    return text