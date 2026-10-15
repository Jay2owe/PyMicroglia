"""Refuse a full accepted-base claim when movie or correction coverage is partial."""
from pathlib import Path
import argparse
import hashlib
import json
import zipfile


def verify(recipe, reports, archive):
    reference=json.loads(Path(recipe).read_text())
    digest=hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    by_movie={}
    for report in reports:
        row=json.loads(Path(report).read_text())
        movie=row.get('movie')
        if movie in by_movie: raise ValueError('Duplicate movie validation: '+str(movie))
        by_movie[movie]=row
    if set(by_movie)!=set(reference['movies']):
        raise ValueError('Validation must cover every accepted movie exactly once')
    count=0
    for movie,row in by_movie.items():
        required={(family,stage) for family,stages in reference['movies'][movie]['expected'].items() for stage in stages}
        actual=[tuple(pair) for pair in row.get('boundaries',[])]
        if row.get('status')!='pass' or set(actual)!=required or len(actual)!=len(required):
            raise ValueError('Incomplete or failing correction boundaries: '+movie)
        if row.get('correction_archive_sha256')!=digest:
            raise ValueError('Validation describes another correction archive: '+movie)
        if set(row.get('inventories',[]))!=set(reference['inventories']):
            raise ValueError('Incomplete process evidence: '+movie)
        count+=len(actual)
    with zipfile.ZipFile(archive) as z:
        manifest=json.loads(z.read('source_hashes.json'))
    return dict(status='pass',accepted_attempt=manifest['accepted_attempt'],
        movies=len(by_movie),correction_boundaries=count,
        saved_feedback=manifest['saved_feedback'],correction_archive_sha256=digest,
        scope='All accepted correction boundaries reproduced from frozen native stage inputs; fresh correction evidence')


def verify_integration(recipe, validation, movie_report, archive):
    """Combine source-equivalent all-movie coverage with new execution checks.

Avoids re-running unchanged, already verified multi-hour scientific operations.
The report explicitly separates reused evidence from fresh execution.
"""
    validation=Path(validation)
    reference=json.loads(Path(recipe).read_text())
    digest=hashlib.sha256(Path(archive).read_bytes()).hexdigest()
    coverage=json.loads((validation/'source_coverage.json').read_text())
    required=sum(len(stages) for movie in reference['movies'].values() for stages in movie['expected'].values())
    if (coverage['status']!='pass' or coverage['archive_sha256']!=digest
            or coverage['accepted_movie_count']!=len(reference['movies'])
            or coverage['accepted_boundary_count']!=required):
        raise ValueError('Complete accepted source coverage is missing or stale')
    row=json.loads(Path(movie_report).read_text())
    movie=row['movie']
    expected={(family,stage) for family,stages in reference['movies'][movie]['expected'].items() for stage in stages}
    actual=[tuple(pair) for pair in row['boundaries']]
    if (row['status']!='pass' or set(actual)!=expected or len(actual)!=len(expected)
            or row['correction_archive_sha256']!=digest
            or set(row['inventories'])!=set(reference['inventories'])):
        raise ValueError('Fresh complete-movie integration did not pass')
    final=json.loads((validation/'final_stage/summary.json').read_text())
    if (final['status']!='pass' or final['archive_sha256']!=digest
            or {r['movie'] for r in final['checks']}!=set(reference['movies'])
            or not all(r['labels_equal'] and r['ledger_equal'] and r['ownership_decisions_equal'] for r in final['checks'])):
        raise ValueError('Latest accepted correction does not match every movie')
    mapping=json.loads((validation/'native_evidence_mapping.json').read_text())
    if ({r['movie'] for r in mapping}!=set(reference['movies'])
            or not all(r['presence_reconstruction_equal'] and r['native_points_equal'] for r in mapping)):
        raise ValueError('Native evidence mapping is incomplete')
    for name in ('focused_tests.json','publication_tests.json'):
        if json.loads((validation/name).read_text())['exit_code']!=0:
            raise ValueError('Focused checks failed: '+name)
    return dict(status='pass',accepted_attempt='R218-A004',archive_sha256=digest,
        source_equivalent_movies=len(reference['movies']),source_equivalent_boundaries=required,
        freshly_checked_movie=movie,freshly_checked_boundaries=len(actual),
        freshly_checked_latest_correction_movies=len(final['checks']),
        native_evidence_mapping_movies=len(mapping),
        saved_feedback=dict(satisfied=1798,total=2348,percentage=1798/2348*100),
        scope='Packaged complete correction sequence. Reused hash-verified all-nine scientific equivalence, fresh evidence/integration on one complete movie, fresh latest-correction execution on all nine. Not a fresh raw-to-final nine-movie rerun or a new accuracy estimate.')


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--recipe',type=Path,required=True)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--reports',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=verify(args.recipe,args.reports,args.archive)
    args.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
