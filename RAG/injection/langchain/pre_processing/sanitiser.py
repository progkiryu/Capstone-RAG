import re
import unicodedata
import warnings
from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning

warnings.filterwarnings("ignore", category=MarkupResemblesLocatorWarning)

HTML_COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
HIDDEN_STYLE = re.compile(r"display\s*:\s*none|visibility\s*:\s*hidden|font-size\s*:\s*0", re.I)
ZERO_WIDTH = re.compile(r"[​-‏⁠-⁤﻿]")
TAG_CHARS = re.compile(r"[\U000E0000-\U000E007F]")


def strip_hidden_html(text):
    soup = BeautifulSoup(HTML_COMMENT.sub(" ", text), "html.parser")
    for element in soup.find_all(style=HIDDEN_STYLE):
        element.decompose()
    for element in soup.find_all(attrs={"hidden": True}):
        element.decompose()
    return soup.get_text(" ")


def sanitise(text):
    text = strip_hidden_html(text)
    text = unicodedata.normalize("NFKC", text)
    text = ZERO_WIDTH.sub("", text)
    text = TAG_CHARS.sub("", text)
    return re.sub(r"\s+", " ", text).strip()
