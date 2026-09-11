"""GSM8K prompt selection and fixed-prompt, real-acceptance BS=1 timing."""
import argparse
import collections
import hashlib
import json
import statistics
import time
from pathlib import Path

import requests

DATA_SHA='3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14'

def sha(data):
    return hashlib.sha256(data).hexdigest()

def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n')

class Client:
    def __init__(self,url):self.url=url.rstrip('/')
    def post(self,route,body=None):
        r=requests.post(self.url+route,json=body or {},timeout=(30,1800));r.raise_for_status();return r
    def wait(self):
        end=time.monotonic()+2400
        while time.monotonic()<end:
            try:
                if requests.get(self.url+'/v1/models',timeout=3).status_code==200:break
            except requests.RequestException:pass
            time.sleep(3)
        else:raise TimeoutError('Server not ready')
        self.post('/freeze_gc')
    def render(self,question):
        messages=[{'role':'user','content':question}]
        body={'model':'deepseek-ai/DeepSeek-V4.1-Flash','messages':messages,
              'reasoning_effort':'high','chat_template_kwargs':{'enable_thinking':True}}
        data=self.post('/v1/tokenize',body).json()
        ids=data['tokens'];assert len(ids)==data['count'] and all(isinstance(i,int) for i in ids)
        return {'messages':messages,'tokenize_request':body,'input_ids':ids,
                'input_ids_sha256':sha(json.dumps(ids,separators=(',',':')).encode())}
    def run(self,prompt,max_tokens=2048):
        self.post('/flush_cache?timeout=30')
        body={'input_ids':prompt['input_ids'],'sampling_params':{
            'temperature':0,'max_new_tokens':max_tokens,'ignore_eos':prompt.get('ignore_eos',False),'stream_interval':1},
            'stream':True}
        start=time.perf_counter();first=None;first_count=None;last=None;last_time=None
        with requests.post(self.url+'/generate',json=body,stream=True,timeout=(30,1800)) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if not line.startswith(b'data: '):continue
                if line[6:]==b'[DONE]':break
                chunk=json.loads(line[6:]);now=time.perf_counter()
                if 'error' in chunk:raise RuntimeError(chunk)
                count=chunk.get('meta_info',{}).get('completion_tokens',0)
                if count and first is None:first=now;first_count=count
                last=chunk;last_time=now
        assert first is not None and last is not None
        meta=last['meta_info'];count=meta['completion_tokens']
        assert meta['prompt_tokens']==len(prompt['input_ids'])
        assert meta.get('cached_tokens',0)==0,meta
        if prompt.get('ignore_eos',False):assert count==max_tokens,meta
        text=last.get('text','')
        words=text.split();ngrams=collections.Counter(tuple(words[i:i+16]) for i in range(max(0,len(words)-15)))
        return {'input_tokens':meta['prompt_tokens'],'output_tokens':count,
            'accept_length':meta.get('spec_accept_length'), 'verify_steps':meta.get('spec_verify_ct'),
            'ttft_s':first-start,'elapsed_s':last_time-start,'first_event_tokens':first_count,
            'output_tps':(count-first_count)/(last_time-first) if last_time>first else None,
            'finish_reason':meta.get('finish_reason'),'max_repeated_16gram':max(ngrams.values(),default=0),
            'request':body,'response':last}

def main():
    p=argparse.ArgumentParser()
    p.add_argument('mode', choices=['bench'])
    p.add_argument('--url', default='http://127.0.0.1:30021')
    p.add_argument('--prompt', type=Path, default=Path(__file__).with_name('prompt.json'))
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--repeat', type=int, default=6)
    p.add_argument('--max-tokens', type=int, default=122)
    a=p.parse_args()
    if a.repeat < 1: p.error('--repeat must be positive')
    a.out.mkdir(parents=True, exist_ok=True)
    c=Client(a.url);c.wait()
    prompt=json.loads(a.prompt.read_text());values=[]
    assert sha(json.dumps(prompt['input_ids'],separators=(',',':')).encode())==prompt['input_ids_sha256']
    for rep in range(a.repeat+1):
        r=c.run(prompt,prompt.get('max_new_tokens',a.max_tokens));r.update(repeat=rep,warmup=rep==0)
        values.append(r);save(a.out/'measurements.json',values)
        print('BENCH',rep,'tps',r['output_tps'],'accept',r['accept_length'],'output',r['output_tokens'],flush=True)
    measured=values[1:]
    summary={'runs':a.repeat,'output_tps_median':statistics.median(r['output_tps'] for r in measured),
      'output_tps_min':min(r['output_tps'] for r in measured),'output_tps_max':max(r['output_tps'] for r in measured),
      'accept_length_median':statistics.median(r['accept_length'] for r in measured) if measured[0]['accept_length'] else None,
      'input_tokens':len(prompt['input_ids']),'output_tokens':[r['output_tokens'] for r in measured],
      'prompt_file_sha256':sha(a.prompt.read_bytes()),'real_acceptance':True}
    save(a.out/'summary.json',summary);print('BENCH_DONE',json.dumps(summary),flush=True)

if __name__=='__main__':main()
