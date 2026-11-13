"""Changed movies as rows; Before / After / Changes as three aligned columns.

Reuse canonical rendered fields at native size, adding change locators and counts.
Never crop fields, rescale images, repeat frames or include unchanged movies.
"""
from pathlib import Path
from contextlib import ExitStack
import argparse,hashlib,json,shutil,tempfile,time,sys
import numpy as np
import tifffile as tf
from PIL import Image,ImageDraw,ImageFont
from scipy import ndimage

HEADER=24
ROW_HEADER=52
GRID_HEADER=64
GUTTER=8
RED=np.array([255,45,45],np.uint8)
YELLOW=np.array([255,230,0],np.uint8)

from functools import lru_cache

@lru_cache(maxsize=32)
def font(size):
    try:result = ImageFont.truetype('arial.ttf',size)
    except OSError:result = ImageFont.load_default(size=size)
    original_mask = getattr(result, 'getmask2', None)
    if original_mask is not None:
        cached_mask = lru_cache(maxsize=2048)(original_mask)
        def mask(*args, **kwargs):
            try:hash((args, tuple(sorted(kwargs.items()))))
            except TypeError:return original_mask(*args, **kwargs)
            return cached_mask(*args, **kwargs)
        result.getmask2 = mask
    return result

def change_mask(field):
    return np.all(field==RED,axis=-1)

def marked_changes(field):
    """Yellow locator boxes only; exact red differences are never enlarged."""
    mask=change_mask(field);out=field.copy()
    groups,n=ndimage.label(ndimage.binary_dilation(mask,iterations=8))
    for box in ndimage.find_objects(groups):
        if box is None:continue
        ys,xs=box;y0,y1=ys.start,ys.stop;x0,x1=xs.start,xs.stop
        out[y0:min(y0+2,y1),x0:x1]=YELLOW;out[max(y0,y1-2):y1,x0:x1]=YELLOW
        out[y0:y1,x0:min(x0+2,x1)]=YELLOW;out[y0:y1,max(x0,x1-2):x1]=YELLOW
    out[mask]=RED
    return out

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(8*1024*1024),b''):h.update(chunk)
    return h.hexdigest()

def layout(sizes):
    if not sizes:raise ValueError('No changed movies to compose')
    width=max(w for h,w in sizes);y=GRID_HEADER;rows=[]
    for h,w in sizes:
        rows.append(y);y+=h+ROW_HEADER+GUTTER
    return dict(columns=3,rows=len(sizes),column_width=width,row_y=rows,
                left_padding=width,right_padding=width,
                height=y-GUTTER,width=width*5+2*GUTTER)

def compose(pages,entries,geometry,frame,frames,attempt):
    canvas=np.zeros((geometry['height'],geometry['width'],3),np.uint8)
    left=geometry['left_padding']
    frame_counts=[]
    for i,(page,entry) in enumerate(zip(pages,entries)):
        h,w=entry['native_shape'];y=geometry['row_y'][i]
        frame_counts.append(int(change_mask(page[HEADER:HEADER+h,2*w:3*w]).sum()))
        for col in range(3):
            x=left+col*(geometry['column_width']+GUTTER)
            canvas[y:y+ROW_HEADER,x:x+geometry['column_width']]=0
            field=page[HEADER:HEADER+h,col*w:(col+1)*w]
            canvas[y+ROW_HEADER:y+ROW_HEADER+h,x:x+w]=marked_changes(field) if col==2 else field
    image=Image.fromarray(canvas);draw=ImageDraw.Draw(image)
    draw.text((left+6,4),f"{attempt} | {len(entries)} movies | frame {frame+1}/{frames} | {sum(frame_counts):,} changed pixels this frame | {sum(e['changed_pixels'] for e in entries):,} across all movies",fill='white',font=font(22))
    draw.text((left+6,34),'RED = exact changed pixels   |   YELLOW BOX = locator only (not extra changes)   |   Movie totals count pixels across all frames',fill=tuple(YELLOW),font=font(18))
    for i,entry in enumerate(entries):
        y=geometry['row_y'][i]
        for col,title in enumerate(['Before','After','Changes']):
            x=left+col*(geometry['column_width']+GUTTER)
            draw.text((x+4,y+3),f"{entry['well']} | {title} | frame {frame+1} | source {frame+entry['source_frame_offset']+1}",fill='white',font=font(18))
            note=f"{frame_counts[i]:,} px this frame | {entry['changed_pixels']:,} px whole movie"
            draw.text((x+4,y+28),note,fill=tuple(YELLOW) if col==2 else 'white',font=font(17))
    output=np.array(image)
    output[:,:left]=0;output[:,-geometry['right_padding']:]=0
    return output

def _build_grid(candidate_manifest,output_dir):
    started=time.perf_counter();candidate_manifest=Path(candidate_manifest).resolve();output_dir=Path(output_dir).resolve()
    candidate=json.loads(candidate_manifest.read_text());entries=[]
    for rec in sorted(candidate['wells'],key=lambda r:-r['changed_pixels']):
        if not rec.get('review_manifest'):continue
        if not rec.get('review_render_decision',{}).get('required',bool(rec['changed_pixels'])):continue
        rp=Path(rec['review_manifest']);rm=json.loads(rp.read_text());item=rm['outputs']['full_field_review'];source=rp.parent/item['relative_path']
        if rm['status']!='complete' or sha(source)!=item['sha256']:raise ValueError('Unverified source review')
        if len(rm['render']['panels'])<3:raise ValueError('Source lacks comparison panels')
        entries.append(dict(well=rec['well'],source=str(source),source_sha256=item['sha256'],review_manifest=str(rp),review_manifest_sha256=sha(rp),native_shape=rm['render']['native_resolution_per_panel'],frames=rm['render']['frames'],source_frame_offset=rm['params']['source_frame_offset'],frame_interval_min=rm['params']['frame_interval_min'],changed_pixels=rec['changed_pixels']))
    if not entries:raise ValueError('No changed reviews; no grid generated')
    if len({q['frames'] for q in entries})!=1 or len({q['frame_interval_min'] for q in entries})!=1:raise ValueError('Movies must share frame count and interval; no truncation or repeated frames allowed')
    geometry=layout([q['native_shape'] for q in entries]);frames=entries[0]['frames']
    fingerprint=hashlib.sha256(json.dumps(dict(entries=entries,layout=geometry,attempt=candidate['attempt'],producer=sha(__file__),execution_helper=sha(Path(__file__).with_name('review_render_speed.py'))),sort_keys=True).encode()).hexdigest()
    if output_dir.exists():
        manifest=json.loads((output_dir/'grid_manifest.json').read_text())
        if manifest['fingerprint']!=fingerprint:raise FileExistsError('Existing grid has different inputs or layout')
        if sha(output_dir/manifest['output']['filename'])!=manifest['output']['sha256']:raise ValueError('Existing grid failed verification')
        return manifest
    peak=frames*geometry['height']*geometry['width']*3+16*1024*1024
    if shutil.disk_usage(output_dir.parent).free<peak:raise OSError('Insufficient space for grid')
    name=f"{candidate['attempt']}_CHANGED_MOVIES_GRID_BEFORE_AFTER_CHANGES_DISPLAY_ONLY.tif"
    with tempfile.TemporaryDirectory(prefix='motion-review-grid-') as temp,ExitStack() as stack:
        sources=[stack.enter_context(tf.TiffFile(q['source'])) for q in entries]
        for movie in sources:
            if len(movie.pages)!=frames:raise ValueError('Incomplete source movie')
        path=Path(temp)/name
        def pages():
            for t in range(frames):yield compose([s.pages[t].asarray() for s in sources],entries,geometry,t,frames,candidate['attempt'])
        shape=(frames,geometry['height'],geometry['width'],3)
        with tf.TiffWriter(path,imagej=peak<4*1024**3,bigtiff=peak>=4*1024**3) as writer:
            writer.write(pages(),shape=shape,dtype=np.uint8,photometric='rgb',compression='zlib',metadata={'axes':'TYXS','finterval':entries[0]['frame_interval_min']*60,'tunit':'sec','unit':'pixel'})
        with tf.TiffFile(path) as result:
            if tuple(result.series[0].shape)!=shape or len(result.pages)!=frames:raise AssertionError('Grid shape mismatch')
            measured=np.zeros((len(entries),frames),dtype=int)
            for t in range(frames):
                page=result.pages[t].asarray()
                if page[:,:geometry['left_padding']].any() or page[:,-geometry['right_padding']:].any():raise AssertionError('Side padding is not black')
                for i,(source,q) in enumerate(zip(sources,entries)):
                    h,w=q['native_shape'];y=geometry['row_y'][i]+ROW_HEADER;original=source.pages[t].asarray()
                    measured[i,t]=int(change_mask(original[HEADER:HEADER+h,2*w:3*w]).sum())
                    for col in range(3):
                        x=geometry['left_padding']+col*(geometry['column_width']+GUTTER)
                        expected=original[HEADER:HEADER+h,col*w:(col+1)*w]
                        if col==2:expected=marked_changes(expected)
                        if not np.array_equal(page[y:y+h,x:x+w],expected):raise AssertionError('Unexpected grid field pixels')
                    x=geometry['left_padding']+2*(geometry['column_width']+GUTTER)
                    if not np.array_equal(change_mask(page[y:y+h,x:x+w]),change_mask(original[HEADER:HEADER+h,2*w:3*w])):raise AssertionError('Red changes differ from source')
            if measured.sum(axis=1).tolist()!=[q['changed_pixels'] for q in entries]:raise AssertionError('Source red pixels disagree with candidate counts')
        manifest=dict(status='complete',fingerprint=fingerprint,attempt=candidate['attempt'],candidate_manifest=str(candidate_manifest),candidate_manifest_sha256=sha(candidate_manifest),producer=str(Path(__file__).resolve()),producer_sha256=sha(__file__),execution_helper_sha256=sha(Path(__file__).with_name('review_render_speed.py')),entries=entries,layout=geometry,frames=frames,panels=['Before','After','Changes'],motion_panel=False,native_pixels_preserved=True,change_locator_overlay='yellow boxes, excluded from pixel counts',per_movie_per_frame_changed_pixels=measured.tolist(),total_changed_pixels=int(measured.sum()),validation='Before/After fields are pixel-exact. Red changed pixels match every source frame exactly. Yellow locator boxes are display-only. Per-frame counts sum to recorded movie totals.',output=dict(filename=name,sha256=sha(path),bytes=path.stat().st_size),seconds=time.perf_counter()-started)
        output_dir.mkdir()
        shutil.copy2(path,output_dir/name)
        if sha(output_dir/name)!=manifest['output']['sha256']:raise AssertionError('Published grid mismatch')
        (output_dir/'grid_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return manifest

def build_grid(candidate_manifest,output_dir):
    from review_render_speed import speed_review_render
    with speed_review_render(sys.modules[__name__]):
        return _build_grid(candidate_manifest,output_dir)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--candidate-manifest',required=True,type=Path);p.add_argument('--output-dir',required=True,type=Path);args=p.parse_args()
    result=build_grid(args.candidate_manifest,args.output_dir)
    print(json.dumps({k:v for k,v in result.items() if k!='entries'},indent=2))
