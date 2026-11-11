with open("app/static/index.html", "r", encoding="utf-8") as f:
    html = f.read()

start = '<script type="text/babel">'
end = '</script>'
p1 = html.find(start)
p2 = html.find(end, p1)
babel_code = html[p1 + len(start):p2]
lines = babel_code.splitlines()

for i, l in enumerate(lines[903:915], start=904):
    print(f"{i}: {l}")

