import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'skills/restore-image-hd/scripts/gpt_segmented_redraw.py'
CLI=ROOT/'tests/fake_image_cli.py'

class PipelineTests(unittest.TestCase):
 def fixture(self,d):
  p=Path(d);im=Image.new('RGB',(128,96),'#8ab1c4');draw=ImageDraw.Draw(im);draw.rectangle((15,10,50,60),fill='white');draw.rectangle((70,30,105,85),fill='green');im.save(p/'source.png')
  for name,box in [('west',(0,0,70,95)),('east',(58,0,127,95))]:
   mask=Image.new('L',(128,96));ImageDraw.Draw(mask).rectangle(box,fill=255);mask.save(p/(name+'.png'))
  (p/'regions.json').write_text(json.dumps({'regions':[{'name':'west','mask':'west.png','priority':2},{'name':'east','mask':'east.png','priority':1}]}))
  return p
 def command(self,p,*args):
  cmd=[sys.executable,str(SCRIPT),str(p/'source.png'),str(p/'out.png'),'--manifest',str(p/'regions.json'),'--work-dir',str(p/'run'),'--gpt-cli',str(CLI),'--mask-expand','0','--context-padding','0','--minimum-padding','0','--alignment','none',*args]
  env={**os.environ,'FAKE_IMAGE_LOG':str(p/'calls.jsonl')}
  return subprocess.run(cmd,capture_output=True,text=True,env=env)
 def test_end_to_end_and_changed_prompt(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.fixture(d);r=self.command(p,'--pilot-count','0');self.assertEqual(r.returncode,0,r.stdout+r.stderr)
   before=(p/'out.png').read_bytes();run=json.loads((p/'run/run.json').read_text())
   self.assertTrue(run['immutable_generation_inputs']);self.assertEqual([r['name'] for r in run['records']],['east','west'])
   self.assertEqual(len((p/'calls.jsonl').read_text().splitlines()),2)
   r=self.command(p,'--pilot-count','0','--resume');self.assertEqual(r.returncode,0,r.stdout+r.stderr)
   self.assertEqual((p/'out.png').read_bytes(),before);self.assertEqual(len((p/'calls.jsonl').read_text().splitlines()),2)
   manifest=json.loads((p/'regions.json').read_text());manifest['regions'][0]['instruction']='changed prompt';(p/'regions.json').write_text(json.dumps(manifest))
   r=self.command(p,'--pilot-count','0','--resume');self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertEqual(len((p/'calls.jsonl').read_text().splitlines()),3)
 def test_pilot_is_real_gate(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.fixture(d);r=self.command(p,'--pilot-count','1','--approve-pilot');self.assertEqual(r.returncode,0,r.stdout+r.stderr)
   self.assertFalse((p/'out.png').exists());self.assertEqual(len((p/'calls.jsonl').read_text().splitlines()),1)
   r=self.command(p,'--pilot-count','1','--resume','--approve-pilot');self.assertEqual(r.returncode,0,r.stdout+r.stderr)
   self.assertTrue((p/'out.png').exists());self.assertEqual(len((p/'calls.jsonl').read_text().splitlines()),2)
 def test_prepare_and_compose_have_no_generation(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.fixture(d);r=self.command(p,'--stage','prepare');self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertFalse((p/'calls.jsonl').exists())
   r=self.command(p,'--stage','compose','--pilot-count','0');self.assertNotEqual(r.returncode,0);self.assertFalse((p/'calls.jsonl').exists())
 def test_oversized_regions_rejected_before_request(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.fixture(d);r=self.command(p,'--target-width','3840','--target-height','2880');self.assertNotEqual(r.returncode,0);self.assertIn('native budget',r.stderr);self.assertFalse((p/'calls.jsonl').exists())
 def test_source_alpha_is_preserved(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.fixture(d);im=Image.open(p/'source.png').convert('RGBA');alpha=Image.new('L',im.size,255);ImageDraw.Draw(alpha).ellipse((0,0,30,30),fill=0);im.putalpha(alpha);im.save(p/'source.png')
   r=self.command(p,'--pilot-count','0');self.assertEqual(r.returncode,0,r.stdout+r.stderr)
   with Image.open(p/'out.png') as out:self.assertEqual(out.getchannel('A').tobytes(),alpha.tobytes())


class StageAndProtectionTests(unittest.TestCase):
 fixture=PipelineTests.fixture
 command=PipelineTests.command
 def test_generate_then_compose_offline(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.fixture(d);r=self.command(p,'--pilot-count','0','--stage','generate');self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertFalse((p/'out.png').exists())
   n=len((p/'calls.jsonl').read_text().splitlines());r=self.command(p,'--pilot-count','0','--stage','compose','--resume');self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertTrue((p/'out.png').exists());self.assertEqual(len((p/'calls.jsonl').read_text().splitlines()),n)
 def test_old_output_is_not_new_pilot_completion(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.fixture(d);r=self.command(p,'--pilot-count','0');self.assertEqual(r.returncode,0,r.stdout+r.stderr)
   manifest=json.loads((p/'regions.json').read_text());manifest['regions'][0]['instruction']='new';(p/'regions.json').write_text(json.dumps(manifest))
   r=self.command(p,'--pilot-count','1','--resume','--approve-pilot');self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertEqual(json.loads((p/'run/status.json').read_text())['state'],'pilot_review')
 def test_edge_pass_does_not_touch_protected_pixels(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.fixture(d);protected=Image.new('L',(128,96));ImageDraw.Draw(protected).rectangle((50,0,60,95),fill=255);protected.save(p/'protected.png')
   r=self.command(p,'--pilot-count','0','--edge-pass','--protect-mask',str(p/'protected.png'));self.assertEqual(r.returncode,0,r.stdout+r.stderr)
   with Image.open(p/'source.png') as src,Image.open(p/'out.png') as out:self.assertEqual(src.crop((50,0,61,96)).tobytes(),out.crop((50,0,61,96)).tobytes())

 def test_selective_retry_keeps_other_regions(self):
  with tempfile.TemporaryDirectory() as d:
   p=self.fixture(d);r=self.command(p,'--pilot-count','0');self.assertEqual(r.returncode,0,r.stdout+r.stderr)
   r=self.command(p,'--pilot-count','0','--resume','--retry-region','west');self.assertEqual(r.returncode,0,r.stdout+r.stderr)
   self.assertEqual(len((p/'calls.jsonl').read_text().splitlines()),3)
   self.assertEqual(len(list((p/'run/jobs').rglob('attempt-*.png'))),3)

if __name__=="__main__":unittest.main()
