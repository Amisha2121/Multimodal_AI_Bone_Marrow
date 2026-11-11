import urllib.request
import uuid

img_path = r'data\derived\CASE-CECDBCC7\fig1_annotated_field.png'
boundary = uuid.uuid4().hex
content_type = f'multipart/form-data; boundary={boundary}'

with open(img_path, 'rb') as f:
    img_bytes = f.read()

parts = [
    f'--{boundary}\r\nContent-Disposition: form-data; name="files"; filename="test_sample.png"\r\nContent-Type: image/png\r\n\r\n'.encode('utf-8'),
    img_bytes,
    f'\r\n--{boundary}--\r\n'.encode('utf-8')
]
body = b''.join(parts)

req = urllib.request.Request(
    'http://127.0.0.1:5000/api/upload',
    data=body,
    headers={'Content-Type': content_type}
)

try:
    with urllib.request.urlopen(req, timeout=40) as r:
        print('Upload Status:', r.status)
        print('Upload Response:', r.read().decode())
except Exception as e:
    print('Upload Error:', e)
    if hasattr(e, 'read'):
        print('Server response:', e.read().decode())

