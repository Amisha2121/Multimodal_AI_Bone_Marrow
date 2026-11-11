"""Quick script to find how depth is encoded in the HTML."""
import re
with open('app/static/index.html', 'r', encoding='utf-8') as f:
    code = f.read()

# data-depth attribute
dd = re.findall(r'data-depth="[^"]*"', code)
print('data-depth occurrences:', len(dd))
for m in dd[:8]:
    print(' ', m)

print()
# className with depth
cn = [line.strip() for line in code.splitlines() if 'depth' in line.lower() and ('className' in line or 'class=' in line)]
print('lines with className+depth:', len(cn))
for m in cn[:8]:
    print(' ', m[:120])

