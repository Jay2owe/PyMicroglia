"""Assemble measured stage costs without adding overlapping worker spans."""
from pathlib import Path
import argparse
import json

REQUIRED=['complete_tracking','native','stage_m0','stage_m1','stage_m2','stage_m3',
          'stage_m4','stage_m5','stage_m7_accepted','stage_m8_accepted',
          'stage_m9_accepted','stage_m12_accepted','stage_m19_accepted',
          'stage_m20_accepted','stage_m22_accepted_history','stage_m6',
          'correction_evidence','corrections','publication_and_cleanup']


def stages(folder):
    rows=[json.loads(line) for line in (folder/'stages.jsonl').read_text().splitlines()]
    starts={};result={}
    samples=[json.loads(line) for line in (folder/'memory.jsonl').read_text().splitlines()]
    for row in rows:
        key=row['stage']
        if row['edge']=='start':starts[key]=row
        elif row['edge']=='end' and key in starts:
            before=starts.pop(key)
            selected=[x['rss'] for x in samples if before['time']<=x.get('time',0)<=row['time']]
            result[key]=dict(seconds=row['time']-before['time'],
                rss_at_start=before['rss'],rss_at_end=row['rss'],
                sampled_stage_peak_rss=max(selected+[before['rss'],row['rss']]) if selected else None,
                cumulative_process_peak_rss=row['peak_rss'],
                cpu_seconds=sum(row.get('cpu',{}).get(k,0)-before.get('cpu',{}).get(k,0) for k in ('user','system')),
                io_delta={k:v-before.get('io',{}).get(k,0) for k,v in row.get('io',{}).items()})
    ends={r['stage']:r for r in rows if r['edge']=='end'}
    if 'corrections' in ends and 'complete_tracking' in ends:
        first,last=ends['corrections'],ends['complete_tracking']
        result['publication_and_cleanup']=dict(seconds=last['time']-first['time'],
            rss_at_start=first['rss'],rss_at_end=last['rss'],
            cumulative_process_peak_rss=last['peak_rss'])
    return dict(stages=result,unfinished=list(starts),summary=json.loads((folder/'summary.json').read_text()))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--before',type=Path,required=True)
    parser.add_argument('--after',type=Path,required=True)
    parser.add_argument('--parity',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    before,after=stages(args.before),stages(args.after)
    parity=json.loads(args.parity.read_text())
    required=REQUIRED
    report=dict(scope='Complete prepared-input tracker, native review, correction evidence, accepted corrections, publication and cleanup',
        notes=['Worker stage spans can overlap: do not sum them as total runtime.',
               'Earlier baseline has boundary RSS only; cumulative process peak is not per-stage peak.',
               'Shared desktop measurements; concurrent unrelated work is not removed from elapsed time.',
               'Process I/O counters do not measure all memory-mapped paging; allocation totals are not physical disk traffic.'],
        before=before,after=after,parity_status=parity['status'],operations=[])
    for key in required:
        left,right=before['stages'].get(key,{}),after['stages'].get(key,{})
        report['operations'].append(dict(id=key,kind='execution',
            status='measured' if left and right else 'unavailable',
            before_seconds=[left['seconds']] if left else [],after_seconds=[right['seconds']] if right else [],
            peak_rss_bytes=after['summary']['peak_working_set_bytes'],
            parity=parity['status']=='pass',receipt=str(args.parity.resolve())))
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    args.output.with_name('speed_memory_coverage_scope.json').write_text(json.dumps({'required':required},indent=2)+'\n')
    print(json.dumps({'before':before['summary'],'after':after['summary'],'parity':parity['status']},indent=2))


if __name__=='__main__':main()
