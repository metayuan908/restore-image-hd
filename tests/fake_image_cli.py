"""Offline image endpoint simulator. Never imports credentials or uses network."""
import argparse,json,os,sys,time
from pathlib import Path
from PIL import Image
if sys.argv[1]=='auth-status':
 print('Offline fake auth available');raise SystemExit(0)
p=argparse.ArgumentParser();p.add_argument('operation');p.add_argument('--image',action='append');p.add_argument('--out');p.add_argument('--background');a,_=p.parse_known_args()
if a.background!='opaque':raise SystemExit('Expected explicit opaque output')
with Image.open(a.image[0]) as opened:im=opened.convert('RGB')
time.sleep(float(os.environ.get('FAKE_IMAGE_DELAY','.02')))
Path(a.out).parent.mkdir(parents=True,exist_ok=True);im.save(a.out)
if os.environ.get('FAKE_IMAGE_LOG'):
 with open(os.environ['FAKE_IMAGE_LOG'],'a') as f:f.write(json.dumps({'crop':a.image[0],'out':a.out,'time':time.time()})+'\n')
