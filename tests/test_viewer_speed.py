"""Speed mechanisms retain native pixels, exact masks and latest selection."""
import base64
from io import BytesIO
import json
from pathlib import Path
import subprocess

import numpy as np
from PIL import Image
import pytest
import tifffile

from pymicroglia.ui import exploration_images as images


def run_fixture(tmp_path, count=5, shape=(300, 420)):
    rng = np.random.default_rng(1733)
    raw = rng.integers(0, 1000, (count+1, *shape), dtype=np.uint16)
    labels = np.zeros((count, *shape), np.uint16)
    labels[:, 3:17, 8:23] = 19
    labels[:, 5:8, 12:15] = 0
    labels[:, 60:66, 70:78] = 26
    for name, value in [('raw', raw), ('labels', labels)]:
        tifffile.imwrite(tmp_path/(name+'.tif'), value, photometric='minisblack', compression='zlib')
    (tmp_path/'manifest.json').write_text(json.dumps({'movies':[{'stem':'test', 'spec':{
        'raw':'raw.tif','labels':'labels.tif','source_frame_offset':1},'display':{'contrast':[0,1000]}}]}))
    return raw, labels


def decoded(value):
    return np.array(Image.open(BytesIO(base64.b64decode(value.split(',')[1]))))


def test_preview_keeps_native_coordinates_and_final_pixels(tmp_path):
    raw, _ = run_fixture(tmp_path)
    before = {p.name:p.read_bytes() for p in tmp_path.iterdir()}
    preview = images.load_frame_image(tmp_path, 'test', 2, preview=True)
    final = images.load_frame_image(tmp_path, 'test', 2)
    assert preview['preview'] and preview['display_only']
    assert (preview['width'], preview['height'], preview['source_frame_index']) == (420,300,3)
    assert max(decoded(preview['image']).shape) == 256
    assert 'preview' not in final
    np.testing.assert_array_equal(decoded(final['image']), (raw[3].astype(float)*255/1000).astype(np.uint8))
    assert before == {p.name:p.read_bytes() for p in tmp_path.iterdir()}
    tifffile.imwrite(tmp_path/'raw.tif', raw+100, photometric='minisblack')
    assert images.load_frame_image(tmp_path, 'test', 2, preview=True)['image'] != preview['image']


def test_small_images_use_native_png_and_route_validates_sources(tmp_path):
    run_fixture(tmp_path, shape=(80,100))
    assert images.load_frame_image(tmp_path,'test',0,preview=True) == images.load_frame_image(tmp_path,'test',0)
    from fastapi.testclient import TestClient
    from pymicroglia.ui.server import create_app
    with TestClient(create_app()) as client:
        params={'run':str(tmp_path),'stem':'test','frame':0,'preview':True}
        assert client.get('/api/exploration/frame-image',params=params).status_code == 200
        for changed in [{'frame':-1},{'frame':999},{'stem':'missing'}]:
            assert client.get('/api/exploration/frame-image',params={**params,**changed}).status_code == 422


@pytest.mark.parametrize('dtype,large', [('uint8',False),('uint16',False),('uint32',False),('uint64',False),('int64',True)])
def test_geometry_preserves_disconnected_pixels_holes_and_large_ids(dtype,large):
    labels = np.zeros((87,113),dtype=dtype)
    identity = 2**40 if large else 19
    labels[1:16,0:30]=identity;labels[5:9,3:11]=0;labels[80:83,101:105]=identity
    labels[40:46,50:60]=26
    expected=[{'identity':int(i),**images.observation(*np.nonzero(labels==i))} for i in np.unique(labels) if i]
    assert images.geometry(labels) == expected
    assert images.geometry(labels,identity) == [row for row in expected if row['identity']==identity]
    assert images.geometry(np.zeros_like(labels)) == []
    with pytest.raises(ValueError,match='nonnegative'):
        images.geometry(-np.ones((2,3),np.int16))


def test_strip_file_opens_are_bounded_and_close_on_error(tmp_path, monkeypatch):
    run_fixture(tmp_path, count=20)
    opened=[]; original=images.tifffile.TiffFile
    def counted(*args,**kwargs):
        handle=original(*args,**kwargs);opened.append(handle);return handle
    monkeypatch.setattr(images.tifffile,'TiffFile',counted)
    assert images.load_cell_images(tmp_path,'test',19)['available']
    assert len(opened)==3  # metadata plus one reader per source, independent of frame count
    assert all(handle.filehandle.closed for handle in opened)
    (tmp_path/'raw.tif').unlink()
    assert not images.load_cell_images(tmp_path,'test',999)['available']
    with pytest.raises(ValueError,match='unavailable'):
        images.load_cell_images(tmp_path,'test',19)
    assert all(handle.filehandle.closed for handle in opened)


def test_progressive_request_races_and_fallbacks():
    script = r'''
    const assert=require('assert'),fs=require('fs'),vm=require('vm');
    const context={window:{}};
    let source=fs.readFileSync('src/pymicroglia/ui/static/exploration.js','utf8');
    source=source.replace('window.PyMicrogliaExploration=','window.TestExplorer=Explorer;window.PyMicrogliaExploration=');
    vm.runInNewContext(source,context);
    const method=context.window.TestExplorer.prototype.fetchFieldMedia;
    const tick=()=>new Promise(resolve=>setImmediate(resolve));
    async function trial(order,{stale=false,failPreview=false,failFull=false}={}){
      const pending={},shown=[];let current=true;
      const e={url:name=>name,options:{api:name=>new Promise((resolve,reject)=>pending[name]={resolve,reject})},
        decodeImage:async image=>({naturalWidth:image==='preview'?256:768}),
        fetchCellMedia:async()=>({change:'exact-change'}),drawImage(){shown.push(this.image.naturalWidth);}};
      const task=method.call(e,1,19,1,{aborted:false},()=>current);
      const handled=task.then(row=>({row}),error=>({error}));
      for(const item of order){
        if(stale)current=false;
        if(item==='preview'){
          if(failPreview)pending['frame-image'].reject(Error('thumbnail failed'));
          else pending['frame-image'].resolve({image:'preview',width:768,height:512});
        }else if(failFull)pending.frame.reject(Error('full failed'));
        else pending.frame.resolve({image:'full',width:768,height:512});
        await tick();
      }
      const result=await handled;
      if(failFull)assert(result.error);else assert.equal(result.row.image.naturalWidth,768);
      assert.deepEqual(shown,stale||failPreview||order[0]==='full'?[]:[256]);
    }
    (async()=>{
      await trial(['preview','full']);await trial(['full','preview']);
      await trial(['preview','full'],{stale:true});await trial(['preview','full'],{failPreview:true});
      await trial(['preview','full'],{failFull:true});await trial(['full','preview'],{failFull:true});
    })().catch(e=>{console.error(e);process.exitCode=1;});
    '''
    subprocess.run(['node','-e',script],cwd=Path(__file__).resolve().parents[1],check=True,timeout=20)
