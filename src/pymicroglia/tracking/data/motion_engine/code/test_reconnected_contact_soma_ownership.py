"""Behavioral checks for witnessed contact ownership, including detector restarts."""
import unittest
import numpy as np
import pandas as pd
from reconnected_contact_soma_ownership import preserve_reconnected_contact_somas


def fixture(restart=True):
    yy,xx=np.indices((36,42));labels=np.zeros((6,36,42),np.uint16)
    raw=np.zeros(labels.shape,np.float32);rows=[]
    for f,x in enumerate([9,10,15,16,16,16]):
        moving=(yy-16)**2+(xx-x)**2<=16
        resident=(yy-16)**2+(xx-25)**2<=16
        labels[f][moving]=7;labels[f][resident]=11
        if f==2:
            labels[f][moving|resident]=11;labels[f,16,x:26]=11
        raw[f]=np.exp(-((yy-16)**2+(xx-x)**2)/8)+2*np.exp(-((yy-16)**2+(xx-25)**2)/8)
        rows += [dict(frame=f,track_id=101 if not restart or f<2 else 202,x=float(x),y=16.,radius_px=2.,strong=True,state='observed'),
                 dict(frame=f,track_id=303,x=25.,y=16.,radius_px=2.,strong=True,state='observed')]
    return labels,raw,pd.DataFrame(rows)


def apply(labels,raw,points,original=None):
    return preserve_reconnected_contact_somas(labels,labels,raw,points,current_labels=labels,
                                              original_labels=labels if original is None else original)


class ContactOwners(unittest.TestCase):
    def test_restart_preserves_both_bodies_without_foreground_change(self):
        labels,raw,points=fixture();after,audit=apply(labels,raw,points)
        self.assertEqual(int(after[2,16,15]),7);self.assertEqual(int(after[2,16,25]),11)
        np.testing.assert_array_equal(after>0,labels>0)
        np.testing.assert_array_equal(after[[0,1,3,4,5]],labels[[0,1,3,4,5]])
        self.assertGreater(int(audit.changed_pixels.sum()),0)
        self.assertEqual(set(np.unique(after)),set(np.unique(labels)))
        self.assertTrue(audit.restart_links.gt(0).any())

    def test_unbroken_measured_path_uses_same_contact_evidence(self):
        labels,raw,points=fixture(False);after,_=apply(labels,raw,points)
        self.assertEqual(int(after[2,16,15]),7)

    def test_missing_contact_observation_is_not_fabricated(self):
        labels,raw,points=fixture();points=points[~((points.frame==2)&(points.track_id==303))]
        after,_=apply(labels,raw,points);np.testing.assert_array_equal(after,labels)

    def test_disagreeing_separate_future_owner_is_not_overwritten(self):
        labels,raw,points=fixture();labels[3:][labels[3:]==7]=13
        after,_=apply(labels,raw,points);np.testing.assert_array_equal(after,labels)

    def test_correct_single_identity_with_two_projections_is_preserved(self):
        labels,raw,points=fixture();labels[labels==7]=11
        after,_=apply(labels,raw,points);np.testing.assert_array_equal(after,labels)

    def test_existing_independent_holder_prevents_duplicate(self):
        labels,raw,points=fixture();labels[2,3:6,3:6]=7
        after,_=apply(labels,raw,points);np.testing.assert_array_equal(after,labels)

    def test_prior_accepted_revision_is_preserved(self):
        labels,raw,points=fixture();original=labels.copy();original[2,16,15]=13
        after,audit=apply(labels,raw,points,original);np.testing.assert_array_equal(after,labels)
        self.assertIn('protected_prior_revision',set(audit.decision))

    def test_third_core_is_not_silently_assigned_to_either_cell(self):
        labels,raw,points=fixture()
        points=pd.concat([points,pd.DataFrame([dict(frame=2,track_id=404,x=20.,y=16.,radius_px=1.,strong=True,state='observed')])],ignore_index=True)
        after,_=apply(labels,raw,points);np.testing.assert_array_equal(after,labels)

    def test_name_and_detector_permutations_do_not_change_decisions(self):
        labels,raw,points=fixture();expected,_=apply(labels,raw,points)
        renamed=labels.copy();renamed[labels==7]=91;renamed[labels==11]=3
        points.track_id=points.track_id.map({101:99,202:2,303:17});points=points.sample(frac=1,random_state=8)
        after,_=apply(renamed,raw,points);restored=after.copy();restored[after==91]=7;restored[after==3]=11
        np.testing.assert_array_equal(restored,expected)

    def test_joint_translation_preserves_exact_partition(self):
        labels,raw,points=fixture();expected,_=apply(labels,raw,points)
        labels=np.pad(labels,((0,0),(7,3),(9,4)));raw=np.pad(raw,((0,0),(7,3),(9,4)))
        points.x+=9;points.y+=7;after,_=apply(labels,raw,points)
        np.testing.assert_array_equal(after,np.pad(expected,((0,0),(7,3),(9,4))))


if __name__=='__main__':unittest.main()
