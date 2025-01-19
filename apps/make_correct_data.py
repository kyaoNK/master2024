import markdown
import re
import csv
from bs4 import BeautifulSoup
from typing import List, Tuple
from pathlib import Path

class RemoveWhitespaceNormalizer:
    def __init__(self) -> None:
        # Define character ranges as class constants
        self.BASIC_LATIN = "\u0000-\u007F"
        self.BLOCKS = "".join((
            "\u4E00-\u9FFF",  # CJK UNIFIED IDEOGRAPHS
            "\u3040-\u309F",  # HIRAGANA
            "\u30A0-\u30FF",  # KATAKANA
            "\u3000-\u303F",  # CJK SYMBOLS AND PUNCTUATION
            "\uFF00-\uFFEF",  # HALFWIDTH AND FULLWIDTH FORMS
        ))
        
        # パターンの追加: 改行文字(\n)と空白文字
        self.patterns = [
            re.compile(f"([{self.BLOCKS}]) ([{self.BLOCKS}])"),
            re.compile(f"([{self.BLOCKS}]) ([{self.BASIC_LATIN}])"),
            re.compile(f"([{self.BASIC_LATIN}]) ([{self.BLOCKS}])"),
            re.compile(r'\n+'),  # 複数の改行を削除
            re.compile(r'\s+')   # 複数の空白文字を1つの空白に置換
        ]

    def normalize(self, text: str) -> str:
        """
        テキストを正規化し、不要な空白と改行を削除する
        """
        # 改行と空白の正規化
        for pattern in self.patterns[:-2]:  # 最初の3つのパターン（日本語・英語の間の空白）
            text = pattern.sub(r"\1\2", text)
        
        # 改行の削除と空白の正規化
        text = self.patterns[-2].sub(' ', text)  # 改行を空白に置換
        text = self.patterns[-1].sub(' ', text)  # 連続する空白を1つに
        return text.strip()  # 前後の空白を削除

def zenkaku_to_hankaku(text: str) -> str:
    """Convert full-width characters to half-width."""
    conversion_map = str.maketrans(
        {chr(0xFF01 + i): chr(0x21 + i) for i in range(94)}
    )
    return text.translate(conversion_map)

def check_figure(text: str) -> bool:
    """Check if text matches figure caption pattern."""
    return bool(re.match(r'^\(図表.+?\)$', text))

def process_markdown_file(input_path: Path, output_path: Path) -> List[Tuple[int, str]]:
    """
    Process markdown file and return numbered paragraphs.
    Args:
        input_path: Path to input markdown file
        output_path: Path to output CSV file
    """
    # 入力ファイルの存在確認
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    # Read and process markdown
    raw_text = input_path.read_text(encoding="utf-8")
    raw_text = zenkaku_to_hankaku(raw_text)
    
    # Parse HTML
    html = markdown.markdown(raw_text)
    soup = BeautifulSoup(html, "html.parser")
    
    # Process paragraphs
    normalizer = RemoveWhitespaceNormalizer()
    materials = []
    count = 0
    
    for element in soup.find_all("p"):
        text = element.text.strip()
        
        if check_figure(text):
            continue
            
        count += 1
        text = normalizer.normalize(text)
        text = zenkaku_to_hankaku(text)
        materials.append([count, text])
    
    # 出力ディレクトリの作成（存在しない場合）
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write to CSV (create new file or overwrite existing one)
    try:
        with output_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["number", "text"])
            writer.writerows(materials)
        print(f"Successfully wrote {len(materials)} rows to {output_path}")
    except Exception as e:
        print(f"Error writing to CSV file: {e}")
        raise
    
    return materials

if __name__ == "__main__":
    input_path = Path("/workspace/data/information1_raw.md")
    output_path = Path("/workspace/out/number_text.csv")
    
    try:
        materials = process_markdown_file(input_path, output_path)
        print(f"Successfully processed {len(materials)} paragraphs")
    except FileNotFoundError as e:
        print(f"Input file error: {e}")
    except Exception as e:
        print(f"Error during processing: {e}")
