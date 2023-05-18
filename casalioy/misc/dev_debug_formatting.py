"""dev utility to debug formatting problems arising in print_HTML"""
from prompt_toolkit import HTML

from casalioy.utils import print_HTML

## Add to print_HTML
# with open("temp.txt", "w", encoding="utf-8") as f:
#     f.write(text.format(**kwargs))

with open("temp.txt", "r", encoding="utf-8") as f:
    s = f.read()

escape_one = lambda v: v.replace("\f", " ").replace("\b", "\\")
s = escape_one(s)

print(s)
print(HTML(s))
print_HTML(s)
