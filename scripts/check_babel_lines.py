import re

with open("app/static/index.html", "r", encoding="utf-8") as f:
    code = f.read()

start_marker = '<script type="text/babel">'
end_marker = '</script>'
p1 = code.find(start_marker)
p2 = code.find(end_marker, p1)
babel_code = code[p1 + len(start_marker):p2]
print("Babel code length:", len(babel_code))

lines = babel_code.splitlines()
print(f"Total lines in Babel script: {len(lines)}")
# The error in screenshot was at line 897 of the inline babel script
for i, l in enumerate(lines[890:905], start=891):
    print(f"{i}: {l}")

