import json,os,sys,tempfile,threading,time,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'skills/restore-image-hd/scripts'))
from PIL import Image,ImageChops,ImageDraw
import numpy as np
from redraw_runtime import run_jobs,signature,file_hash,JobLock,native_budget,save_capacity
from redraw_quality import inspect_native,normalize_native,align_donor,minimum_error_selector
from gpt_segmented_redraw import composite_region

class RuntimeTests(unittest.TestCase):
 def jobs(self,n=4):return [{'id':str(i),'signature':signature({'i':i})} for i in range(n)]
 def inspect(self,p,j):
  with Image.open(p) as im:return {'native_dimensions':list(im.size)}
 def test_parallel_order_resume_and_hash(self):
  with tempfile.TemporaryDirectory() as d:
   count=[];active=0;peak=0;lock=threading.Lock()
   def generate(j,p):
    nonlocal active,peak
    with lock:active+=1;peak=max(peak,active);count.append(j['id'])
    time.sleep(.04);Image.new('RGB',(32,32)).save(p)
    with lock:active-=1
   rows=run_jobs(self.jobs(),generate,self.inspect,d,workers=2)
   self.assertEqual([r['job'] for r in rows],['0','1','2','3']);self.assertEqual(peak,2)
   rows=run_jobs(self.jobs(),generate,self.inspect,d,workers=2,resume=True)
   self.assertEqual(len(count),4);self.assertTrue(all(r['resumed'] for r in rows))
   changed=self.jobs();changed[1]['signature']=signature({'changed':1})
   run_jobs(changed,generate,self.inspect,d,workers=2,resume=True)
   self.assertEqual(len(count),5)
   Path(rows[0]['raw_path']).write_bytes(b'changed')
   with self.assertRaisesRegex(RuntimeError,'missing or changed'):run_jobs(self.jobs(),generate,self.inspect,d,resume=True)
 def test_partial_failure_resume(self):
  with tempfile.TemporaryDirectory() as d:
   calls=[]
   def gen(j,p):
    calls.append(j['id'])
    if j['id']=='1':raise RuntimeError('connection dropped')
    Image.new('RGB',(32,32)).save(p)
   with self.assertRaisesRegex(RuntimeError,'resume'):run_jobs(self.jobs(2),gen,self.inspect,d,workers=2)
   def fixed(j,p):calls.append(j['id']);Image.new('RGB',(32,32)).save(p)
   run_jobs(self.jobs(2),fixed,self.inspect,d,workers=2,resume=True)
   self.assertEqual(calls.count('0'),1);self.assertEqual(calls.count('1'),2)
 def test_rate_limit_degrades_dispatch(self):
  with tempfile.TemporaryDirectory() as d:
   calls={};active=0;peak_after=0;limited=threading.Event();lock=threading.Lock()
   def gen(j,p):
    nonlocal active,peak_after
    with lock:
     calls[j['id']]=calls.get(j['id'],0)+1;active+=1
     if limited.is_set():peak_after=max(peak_after,active)
    try:
     if j['id']=='0' and calls['0']==1:
      limited.set();raise RuntimeError('HTTP 429 too many requests')
     time.sleep(.03);Image.new('RGB',(32,32)).save(p)
    finally:
     with lock:active-=1
   rows=run_jobs(self.jobs(),gen,self.inspect,d,workers=2,retries=1,retry_delay=0)
   self.assertEqual(len(rows),4);self.assertEqual(calls['0'],2)
   events=[json.loads(s) for s in (Path(d)/'events.jsonl').read_text().splitlines()]
   self.assertTrue(any(e['stage']=='concurrency_reduced' for e in events))
 def test_lock(self):
  with tempfile.TemporaryDirectory() as d:
   with JobLock(d):
    with self.assertRaises(RuntimeError):
     with JobLock(d):pass
 def test_capacity_scope_and_expiry(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'profile.json';save_capacity(p,[{'native_dimensions':[1500,1000]}],'scope')
   self.assertEqual(native_budget(p,'scope'),1200000)
   with self.assertRaisesRegex(ValueError,'different'):native_budget(p,'other')
   j=json.loads(p.read_text());j['observed_at']=0;p.write_text(json.dumps(j))
   with self.assertRaisesRegex(ValueError,'expired'):native_budget(p,'scope')

class QualityTests(unittest.TestCase):
 def pattern(self):
  im=Image.new('RGB',(256,256),'#8ab1c4');d=ImageDraw.Draw(im)
  for x,y,w,h in [(25,30,70,50),(120,30,50,90),(50,140,110,50),(180,150,40,70)]:
   d.rectangle((x,y,x+w,y+h),fill='white',outline='black',width=3)
  return im
 def test_alpha_and_native_size(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'a.png';Image.new('RGBA',(200,200),(0,0,0,0)).save(p)
   with self.assertRaisesRegex(RuntimeError,'transparent'):inspect_native(p,(100,100))
   Image.new('RGB',(80,80)).save(p)
   with self.assertRaisesRegex(RuntimeError,'below'):inspect_native(p,(100,100))
   Image.new('RGB',(200,190)).save(p)
   with self.assertRaisesRegex(RuntimeError,'aspect'):inspect_native(p,(100,100))
 def test_translation_corrects_without_bending(self):
  base=self.pattern();donor=base.transform(base.size,Image.Transform.AFFINE,(1,0,-7,0,1,4))
  result,report=align_donor(base,donor,'translation',12,5)
  self.assertTrue(report['applied']);self.assertLess(report['after']['p95_pixels'],report['before']['p95_pixels'])
  a=np.array(base)[15:-15,15:-15];b=np.array(result)[15:-15,15:-15]
  self.assertLess(float(np.abs(a.astype(float)-b).mean()),1)
 def test_large_shift_rejected(self):
  base=self.pattern();donor=base.transform(base.size,Image.Transform.AFFINE,(1,0,-40,0,1,0))
  with self.assertRaisesRegex(RuntimeError,'exceeds'):align_donor(base,donor,'translation',10,5)
 def test_mask_protection_is_exact_after_feather(self):
  base=Image.new('RGB',(100,100),'red');mask=Image.new('L',(100,100));ImageDraw.Draw(mask).rectangle((20,20,80,80),fill=255)
  composite_region(base,Image.new('RGB',(100,100),'blue'),mask,(0,0,100,100),10)
  self.assertEqual(base.getpixel((19,50)),(255,0,0));self.assertNotEqual(base.getpixel((50,50)),(255,0,0))
 def test_seam_finds_continuous_low_error_path(self):
  old=Image.new('RGB',(100,80),(40,40,40));new=Image.new('RGB',(100,80),(90,90,90))
  d=ImageDraw.Draw(new);d.rectangle((48,0,50,79),fill=(40,40,40))
  mask=Image.new('L',(100,80));ImageDraw.Draw(mask).rectangle((35,0,99,79),fill=255)
  cover=Image.new('L',(100,80));ImageDraw.Draw(cover).rectangle((0,0,65,79),fill=255)
  selector,report=minimum_error_selector(old,new,mask,cover)
  self.assertTrue(report['used']);self.assertEqual(report['mean_path_error'],0)
  self.assertEqual(selector.getpixel((40,40)),0);self.assertEqual(selector.getpixel((60,40)),255)

if __name__=='__main__':unittest.main()
