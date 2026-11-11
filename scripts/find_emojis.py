import os
import re

emoji_pattern = re.compile(
    r'[\U00010000-\U0010ffff]|[\u2600-\u27bf]|[\u2300-\u23ff]|[\u2b50-\u2b55]|[\u203c-\u2049]|[\u25aa-\u25fe]'
)

target_dirs = ['app', 'pipeline', 'agents']
findings = []

for td in target_dirs:
    for root, dirs, files in os.walk(td):
        for f in files:
            if f.endswith(('.html', '.py', '.js', '.css')):
                fpath = os.path.join(root, f)
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
                    for lno, line in enumerate(fp, 1):
                        matches = emoji_pattern.findall(line)
                        if matches:
                            findings.append((fpath, lno, ''.join(set(matches)), line.strip()))

print(f"Total emoji lines: {len(findings)}")
for fpath, lno, emojis, line in findings:
    print(f"{fpath}:{lno} [{emojis}] {line[:90]}")

