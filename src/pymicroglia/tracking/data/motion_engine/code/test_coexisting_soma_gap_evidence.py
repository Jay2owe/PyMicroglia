"""Boundary checks for physical coexistence evidence, not label correctness."""
import unittest
import numpy as np
import pandas as pd
from coexisting_soma_gap_evidence import CoexistingSomaGapEvidence


class CoexistenceTests(unittest.TestCase):
    def scene(self):
        raw=np.full((5,40,50),10.,dtype=float)
        yy,xx=np.indices(raw.shape[1:]);masks=[];rows=[]
        for track,y,x in [(4,13,12),(91,25,35)]:
            masks.append((abs(yy-y)<=4)&(abs(xx-x)<=4))
            raw += 100*np.exp(-((yy-y)**2+(xx-x)**2)/5.)
            rows.extend(dict(frame=f,track_id=track,x=x,y=y,strong=True,state='observed') for f in range(5))
        return raw,pd.DataFrame(rows),masks

    def test_independent_bodies_and_reverse_direction(self):
        raw,points,masks=self.scene();e=CoexistingSomaGapEvidence(raw,points)
        for a,b in [(0,4),(4,0)]:
            result=e.measure(masks[0],a,masks[1],b)
            self.assertTrue(result['coexisting_raw_somas']);self.assertEqual(result['frames_checked'],5)

    def test_missing_signal_or_measurement_does_not_prove_coexistence(self):
        raw,points,masks=self.scene();raw[2][masks[0]]=10
        self.assertFalse(CoexistingSomaGapEvidence(raw,points).measure(masks[0],0,masks[1],4)['coexisting_raw_somas'])
        raw,points,masks=self.scene();points=points[~((points.frame==2)&(points.track_id==4))]
        self.assertFalse(CoexistingSomaGapEvidence(raw,points).measure(masks[0],0,masks[1],4)['coexisting_raw_somas'])

    def test_protected_adjacent_same_core_and_shared_outline(self):
        raw,points,masks=self.scene();e=CoexistingSomaGapEvidence(raw,points)
        for source,start,target,end in [(masks[0],0,masks[1],1),(masks[0],0,masks[0],4),(masks[0]|masks[1],0,masks[1],4)]:
            self.assertFalse(e.measure(source,start,target,end)['coexisting_raw_somas'])

    def test_joint_translation_and_bijective_core_renaming(self):
        raw,points,masks=self.scene();expected=CoexistingSomaGapEvidence(raw,points).measure(masks[0],0,masks[1],4)
        raw=np.pad(raw,((0,0),(3,0),(7,0)),constant_values=10)
        points.x+=7;points.y+=3;points.track_id=points.track_id.map({4:803,91:2})
        masks=[np.pad(m,((3,0),(7,0))) for m in masks]
        actual=CoexistingSomaGapEvidence(raw,points).measure(masks[0],0,masks[1],4)
        for key in ('coexisting_raw_somas','frames_checked','reason'):self.assertEqual(actual[key],expected[key])


if __name__=='__main__':unittest.main()
