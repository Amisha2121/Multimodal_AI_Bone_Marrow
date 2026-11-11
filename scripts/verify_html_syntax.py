import re

with open('app/static/index.html', 'r', encoding='utf-8') as f:
    text = f.read()

m = re.search(r'<script type="text/babel">(.*?)</script>', text, re.DOTALL)
if not m:
    print("NO BABEL SCRIPT FOUND")
else:
    code = m.group(1)
    open_b = code.count('{')
    close_b = code.count('}')
    print(f"Braces: {open_b} open, {close_b} close, balance = {open_b - close_b}")
    open_p = code.count('(')
    close_p = code.count(')')
    print(f"Parentheses: {open_p} open, {close_p} close, balance = {open_p - close_p}")
    open_br = code.count('[')
    close_br = code.count(']')
    print(f"Brackets: {open_br} open, {close_br} close, balance = {open_br - close_br}")

    # Check for raw less thans
    lines = code.split('\n')
    for idx, l in enumerate(lines):
        matches = re.findall(r'(?<![=\{`\'"<a-zA-Z/!])<\d+', l)
        if matches:
            print(f"Potential raw less than at line {idx+1}: {l.strip()}")
print("Syntax check finished.")

