import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];SCRIPTS=ROOT/'skills/restore-image-hd/scripts';CLI=ROOT/'tests/fake_image_cli.py'
class RouteTests(unittest.TestCase):
 def run_route(self,route,p,*extra):
  env={**os.environ,'FAKE_IMAGE_LOG':str(p/'calls.jsonl')}
  args=[sys.executable,str(SCRIPTS/route),str(p/'source.png'),str(p/'out.png'),'--content-mode','scenery' if route=='gpt_tile_redraw.py' else 'architecture','--target-width','256','--target-height','192','--gpt-cli',str(CLI),'--work-dir',str(p/'job'),'--pilot-count','0','--alignment','none',*extra]
  return subprocess.run(args,capture_output=True,text=True,env=env)
 def test_tile_and_coherent_routes_use_durable_engine(self):
  for route in ['gpt_tile_redraw.py','gpt_coherent_redraw.py']:
   with self.subTest(route=route),tempfile.TemporaryDirectory() as d:
    p=Path(d);im=Image.new('RGB',(128,96),'#89a5bf');draw=ImageDraw.Draw(im);draw.rectangle((20,20,60,65),fill='white');im.save(p/'source.png')
    r=self.run_route(route,p,'--tile-width','128','--tile-height','128','--overlap-x','32','--overlap-y','32')
    self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertTrue((p/'out.png').is_file())
    n=len((p/'calls.jsonl').read_text().splitlines());self.assertGreater(n,1)
    before=(p/'out.png').read_bytes();r=self.run_route(route,p,'--tile-width','128','--tile-height','128','--overlap-x','32','--overlap-y','32','--resume')
    self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertEqual(len((p/'calls.jsonl').read_text().splitlines()),n);self.assertEqual(before,(p/'out.png').read_bytes())
 def test_dry_run_never_generates(self):
  with tempfile.TemporaryDirectory() as d:
   p=Path(d);Image.new('RGB',(128,96)).save(p/'source.png');r=self.run_route('gpt_coherent_redraw.py',p,'--dry-run');self.assertEqual(r.returncode,0,r.stdout+r.stderr);self.assertFalse((p/'calls.jsonl').exists())
if __name__=='__main__':unittest.main()
