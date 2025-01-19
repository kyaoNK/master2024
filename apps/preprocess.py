import markdown
import re
from bs4 import BeautifulSoup

class RemoveWhitespaceNormalizer:
    basic_latin = "\u0000-\u007F"
    blocks = "".join(
        (
            "\u4E00-\u9FFF",  # CJK UNIFIED IDEOGRAPHS
            "\u3040-\u309F",  # HIRAGANA
            "\u30A0-\u30FF",  # KATAKANA
            "\u3000-\u303F",  # CJK SYMBOLS AND PUNCTUATION
            "\uFF00-\uFFEF",  # HALFWIDTH AND FULLWIDTH FORMS
        )
    )

    def __init__(self) -> None:
        pattern1 = re.compile("([{}]) ([{}])".format(self.blocks, self.blocks))
        pattern2 = re.compile(
            "([{}]) ([{}])".format(self.blocks, self.basic_latin)
        )
        pattern3 = re.compile(
            "([{}]) ([{}])".format(self.basic_latin, self.blocks)
        )
        self.patterns = (pattern1, pattern2, pattern3)

    def normalize(self, text: str) -> str:
        for pattern in self.patterns:
            while pattern.search(text):
                text = pattern.sub(r"\1\2", text)
        return text

def zenkaku_to_hankaku(text):
    # Full-width digit range
    text = re.sub(r'[０-９]', lambda x: chr(ord(x.group(0)) - 0xFEE0), text)
    # Range of full-width alphabetic characters (upper and lower case)
    text = re.sub(r'[Ａ-Ｚａ-ｚ]', lambda x: chr(ord(x.group(0)) - 0xFEE0), text)
    return text

def check_figure(text):
    pattern = r'^\（図表.+?\）$'
    if re.match(pattern, text):
        return True
    else:
        return False

def preprocessing(raw_filepath: str):
    with open(raw_filepath, "r", encoding="utf-8") as raw_file:
        raw_text = raw_file.read()

    raw_text = zenkaku_to_hankaku(raw_text)

    html = markdown.markdown(raw_text)

    soup = BeautifulSoup(html, "html.parser")

    elements = soup.find_all("p")

    count = 0
    materials = dict()

    normalizer = RemoveWhitespaceNormalizer()

    for element in elements:
        element_only_text = element.text.strip()

        if check_figure(element_only_text):
            continue

        element_only_text = normalizer.normalize(element_only_text)
        element_only_text = zenkaku_to_hankaku(element_only_text)

        count += 1
        materials[count] = {"label": "material", "text": element_only_text}
    return materials

if __name__=='__main__':
    raw_filepath = "data/information1_1_raw.md"
    materials = preprocessing(raw_filepath)

    for key, value in materials.items():
        print(f"{key}: {value}")