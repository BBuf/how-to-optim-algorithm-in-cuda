"""Pinned legacy five-shot GSM8K prompts/scorer, with complete response/error records."""
import argparse,hashlib,json,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import requests
from sglang.test.simple_eval_mixed_prefix_gsm8k import get_few_shot_examples,get_one_example,get_answer_value

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--threads',type=int,required=True);p.add_argument('--count',type=int,default=1314);p.add_argument('--url',default='http://127.0.0.1:30021');a=p.parse_args()
 a.out.mkdir(exist_ok=True);data=Path(__file__).with_name('gsm8k-test.jsonl');rows=[json.loads(l) for l in data.read_text().splitlines()];assert len(rows)==1319
 prefix=get_few_shot_examples(rows,5);ids=list(range(5,min(1319,5+a.count)))
 models=requests.get(a.url+'/v1/models',timeout=30);models.raise_for_status();model=models.json()['data'][0]['id']
 def run(i):
  prompt=prefix+get_one_example(rows,i,False);start=time.perf_counter()
  rec=dict(index=i,prompt=prompt,prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),expected=get_answer_value(rows[i]['answer']))
  try:
   r=requests.post(a.url+'/v1/chat/completions',json=dict(model=model,messages=[dict(role='user',content=prompt)],temperature=0,top_p=1,max_tokens=4096,seed=0,return_meta_info=True),timeout=(30,1800));r.raise_for_status();j=r.json();rec['response']=j
   ch=j['choices'][0];content=ch['message'].get('content') or '';rec.update(answer=get_answer_value(content),correct=get_answer_value(content)==rec['expected'],truncated=ch['finish_reason']=='length',empty=not bool(content.strip()))
  except Exception as e:rec.update(error=repr(e),correct=False,truncated=False,empty=True)
  rec['elapsed_s']=time.perf_counter()-start;return rec
 results=[];begin=time.perf_counter()
 with (a.out/'samples.jsonl').open('w') as f,ThreadPoolExecutor(max_workers=a.threads) as pool:
  futures=[pool.submit(run,i) for i in ids]
  for future in as_completed(futures):
   r=future.result();results.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
   if len(results)%25==0 or len(results)==len(ids):print('GSM',len(results),'/',len(ids),'correct',sum(x['correct'] for x in results),'errors',sum('error' in x for x in results),flush=True)
 summary=dict(count=len(results),correct=sum(r['correct'] for r in results),score=sum(r['correct'] for r in results)/len(results),errors=sum('error' in r for r in results),truncated=sum(r['truncated'] for r in results),empty=sum(r['empty'] for r in results),threads=a.threads,elapsed_s=time.perf_counter()-begin,dataset_sha256=hashlib.sha256(data.read_bytes()).hexdigest(),num_shots=5,held_out_indices=ids,temperature=0,top_p=1,seed=0,max_tokens=4096)
 (a.out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
 assert summary['errors']==0,summary
 assert summary['score']>=.9,summary
if __name__=='__main__':main()
