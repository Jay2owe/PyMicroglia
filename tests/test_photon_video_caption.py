"""Photon exports use the current shared caption API and recorded timestamps."""
import numpy as np


def test_all_photon_streams_preserve_pixels_below_caption_and_use_elapsed_time(tmp_path,monkeypatch):
    from pymicroglia.pipelines import cry1_dluc_photon as pipeline
    original=pipeline._annotate.stamp
    captions=[];streams=[]
    def checked_stamp(rgb,text,**options):
        result=original(rgb,text,**options)
        assert result.shape==(rgb.shape[0]+48,rgb.shape[1],3)
        np.testing.assert_array_equal(result[48:],rgb)
        captions.append(text)
        return result
    monkeypatch.setattr(pipeline._annotate,'stamp',checked_stamp)
    def consume(path,frames,**options):streams.append(list(frames))
    monkeypatch.setattr(pipeline._encode,'write_video',consume)
    pixels=np.random.default_rng(1).integers(1,1000,(3,4,32,48),dtype=np.uint16)
    positive=pixels[:,0].astype('float32')
    settings={'short_frames':3,'long_frames':5,
              **{key:[0.,100.] for key in ('brightfield_display_percentiles','neuron_display_percentiles',
                       'photon_raw_display_percentiles','photon_anscombe_display_percentiles')}}
    written,_=pipeline._videos(tmp_path,pixels,pixels,positive,positive,positive,
        np.array([0.,1800.,5400.]),(0,0,48,32),fps=4.,crf=18,settings=settings)
    assert len(written)==len(streams)==4
    assert all(len(frames)==3 for frames in streams)
    assert [text.rsplit('|',1)[-1].strip() for text in captions]==['00:00:00','00:30:00','01:30:00']*4
