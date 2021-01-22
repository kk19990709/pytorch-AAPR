from torchtext.data import Field
import nltk
import re

# nltk.download(['wordnet', "stopwords", "tagsets", "averaged_perceptron_tagger", "punkt", "vader_lexicon"])
english_stopwords = nltk.corpus.stopwords.words("english")

def tokenizer_author(text):
    return text.split(',')

def tokenizer_category(text):
    return text.split(',')


def clean_math(string):
    while string.count('$') > 1:
        pos0 = string.find('$')
        pos1 = string.find('$', pos0+1)
        string = (string[:pos0] + string[pos1+1:]).strip()
    return string


def clean_str(string):
    """
    Input:
        string: One line in a latex file.
    Return：
        string cleaned.
    """
    # Remove mathematical formulas between $$
    string = clean_math(string)

    # Remove "ref"
    string = re.sub(r'~(.*)}', '', string)
    string = re.sub(r'\\cite(.*)}', '', string)
    string = re.sub(r'\\newcite(.*)}', '', string)
    string = re.sub(r'\\ref(.*)}', '', string)

    # Remove stopwords
    texts_tokenized = [word.lower() for word in nltk.tokenize.word_tokenize(string)]
    texts_filtered_stopwords = [word for word in texts_tokenized if not word in english_stopwords]
    string = ' '.join(texts_filtered_stopwords)
    string = string.replace(',', '')
    string = string.replace('.', '')
    string = string.replace('?', '')
    string = string.replace('!', '')
    string = string.replace('/', '')
    string = string.replace('$', '')
    string = string.replace('~', '')
    string = string.replace('\\', '')
    string = string.replace('{', '')
    string = string.replace('}', '')
    string = string.replace('#', '')
    string = string.replace('&', '')
    string = string.replace('@', '')
    string = string.replace('%', '')
    string = string.replace('^', '')
    string = string.replace('*', '')
    string = string.replace('-', '')
    string = string.replace('=', '')
    string = string.replace('[', '')
    string = string.replace(']', '')
    string = string.replace('+', '')
    string = string.replace('(', '')
    string = string.replace(')', '')
    return string


def process_text_list(text_list):
    """
    Input:
        text_list: Content of a latex file and each element represents a line.
    Return:
        A list, which is the cleaned content of a latex file.
    """

    result = ''
    for line in text_list:
        line = line.strip()
        if line.startswith('%') or line.startswith('\\') or line == '':  # TODO 去掉注释，但是\\不知道什么时候用
            pass
        elif line[0].isdigit():  # TODO 不知道为什么要去掉数字
            pass
        else:
            result += clean_str(line)
    return result


# Extract Introduction, related work, etc.================================================================
def extract(text):
    text = ' '.join(text)
    sections = []
    for section in ('Intro', 'Related', 'Conclu'):
        temp = re.search(r'\\Section({' + section + '.*?)\\Section', text, flags=re.S|re.I)
        if temp is None:
            sections.append('')
        else:
            sections.append(temp.group(1))
    intro, related, conclu = sections
    methods = text.replace(intro, '').replace(related, '').replace(conclu, '')
    return list(map(process_text_list, [intro.split('\n'), related.split('\n'), methods.split('\n'), conclu.split('\n')]))