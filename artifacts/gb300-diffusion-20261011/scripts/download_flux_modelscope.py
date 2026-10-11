"""Download only the pinned, HF-equivalent Diffusers components from ModelScope."""
from concurrent.futures import ThreadPoolExecutor
from http.client import IncompleteRead
import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import HTTPError
import time
from urllib.parse import quote

ROOT=Path('/scratch/bbuf-diffusion-20261011')
manifest=json.loads((ROOT/'flux-modelscope-verified-manifest.json').read_text())
cache=ROOT/'models/flux1'
snapshot=cache/'snapshot'
snapshot.mkdir(parents=True,exist_ok=True)

def sha256_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''): h.update(chunk)
    return h.hexdigest()

def download(row):
    dest=snapshot/row['path']
    dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.exists() and dest.stat().st_size==row['size'] and sha256_file(dest)==row['sha256']:
        return row['path']
    url='https://modelscope.cn/models/black-forest-labs/FLUX.1-dev/resolve/'+row['ms_revision']+'/'+quote(row['path'])
    partial=dest.with_suffix(dest.suffix+'.partial')
    for attempt in range(5):
        try:
            offset=partial.stat().st_size if partial.exists() else 0
            if offset>row['size']:
                partial.unlink();offset=0
            if offset<row['size']:
                headers={'User-Agent':'Mozilla/5.0'}
                if offset:headers['Range']=f'bytes={offset}-'
                with urlopen(Request(url,headers=headers),timeout=120) as response:
                    if offset and response.status==206:
                        content_range=response.headers.get('Content-Range','')
                        assert content_range.startswith(f'bytes {offset}-'),content_range
                        mode='ab'
                    else:
                        assert response.status==200,response.status
                        mode='wb'
                    with partial.open(mode) as out:
                        for chunk in iter(lambda:response.read(8*1024*1024),b''):
                            out.write(chunk)
            assert partial.stat().st_size==row['size'],row['path']
            digest=sha256_file(partial)
            if digest!=row['sha256']:
                partial.unlink()
                raise ValueError(f"SHA256 mismatch: {row['path']}")
            if row['hf_lfs_sha256']:assert digest==row['hf_lfs_sha256']
            else:
                contents=partial.read_bytes()
                assert hashlib.sha1(b'blob '+str(len(contents)).encode()+b'\0'+contents).hexdigest()==row['hf_git_blob']
            break
        except HTTPError as exc:
            if exc.code not in (408,429,500,502,503,504) or attempt==4:raise
            print('RETRY',row['path'],type(exc).__name__,attempt+1,flush=True)
            time.sleep(min(30,2**attempt))
        except (OSError,IncompleteRead,AssertionError,ValueError) as exc:
            if attempt==4:raise
            print('RETRY',row['path'],type(exc).__name__,attempt+1,flush=True)
            time.sleep(min(30,2**attempt))
    partial.replace(dest)
    print('VERIFIED',row['path'],row['size'],flush=True)
    return row['path']

checkpoint=dict(model=manifest['model'],revision=manifest['hf_revision'],task_owned_cache=str(cache),snapshot_path=str(snapshot),source=manifest['source'],file_manifest=str(ROOT/'flux-modelscope-verified-manifest.json'))
out=ROOT/'results/flux1-checkpoint.json'
out.write_text(json.dumps(checkpoint,indent=2)+'\n')
with ThreadPoolExecutor(max_workers=4) as pool: downloaded=list(pool.map(download,manifest['files']))
checkpoint.update(download_complete=True,verified_files=len(downloaded),downloaded_bytes=manifest['total_bytes'])
out.write_text(json.dumps(checkpoint,indent=2)+'\n')
print(json.dumps(checkpoint,indent=2),flush=True)
