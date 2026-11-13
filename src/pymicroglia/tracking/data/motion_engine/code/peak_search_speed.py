"""Exact greedy peak separation using nearby integer spatial buckets."""
from contextlib import contextmanager
import math


@contextmanager
def bucketed_peak_search(module):
    original=module._greedy_separated_peaks
    def separated(evidence,candidates,minimum_distance):
        ordered=sorted(((float(evidence[y,x]),int(y),int(x)) for y,x in candidates),reverse=True)
        distance=float(minimum_distance);distance2=distance**2
        if not math.isfinite(distance2):return original(evidence,candidates,minimum_distance)
        if distance2==0:return [(y,x) for _,y,x in ordered]
        # Integer bucket width is at least the unchanged separation distance.
        # Therefore any rejected pair must occupy this or an adjacent bucket.
        width=max(1,math.ceil(abs(distance)));buckets={};accepted=[]
        for _,y,x in ordered:
            by,bx=y//width,x//width
            nearby=(point for iy in range(by-1,by+2) for ix in range(bx-1,bx+2) for point in buckets.get((iy,ix),()))
            if all((y-ay)**2+(x-ax)**2>=distance2 for ay,ax in nearby):
                accepted.append((y,x));buckets.setdefault((by,bx),[]).append((y,x))
        return accepted
    module._greedy_separated_peaks=separated
    try:yield
    finally:module._greedy_separated_peaks=original
