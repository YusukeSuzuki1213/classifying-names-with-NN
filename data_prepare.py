import torch
import glob
import unicodedata
import string
import os

def findFiles(path): return glob.glob(path)

# Unicode -> ASCII
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn' and c in all_letters
    )

# ファイルを読み込む
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# all_lettersからletterのindexを探す,  e.g. "a" = 0
def letterToIndex(letter):
    return all_letters.find(letter)

# letterを1×n_lettersのone-hotベクトルに変換
def letterToTensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letterToIndex(letter)] = 1
    return tensor

# lineをletter×1×n_lettersのoneotベクトルの行列に変換
def lineToTensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letterToIndex(letter)] = 1
    return tensor

all_letters = string.ascii_letters + " .,;'-" 
n_letters = len(all_letters)

category_lines = {} # Type: Dict[str,List[str]], e.g. {'Italian': ['Abandonato', 'Abatangelo',...]}
all_categories = [] # Type: List[str], e.g. ['data/names/French.txt', 'data/names/Czech.txt',...]

# category_lines, all_categoriesを作成
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)
