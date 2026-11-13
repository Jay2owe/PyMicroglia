"""Original final ownership operations, supplied with their native evidence.

Native evidence must follow the upstream run being evaluated. This module does
not select recordings, identities or feedback and does not create that evidence.
"""
import sys
from recovery_sequence_composition import frozen_method


def apply(labels,ledger,evidence,operations, *, source_frame_offset=0):
    for spec in operations:
        data=evidence[spec['stage']]
        with frozen_method(spec) as method:
            if spec['kind']=='gap':
                candidate,audit=method(labels,data['original_labels'],data['points'],data['traces'],source_frame_offset=source_frame_offset)
                tables=dict(source_branch_audit=audit)
            elif spec['kind']=='restoration':
                candidate,audit=method(labels,data['original_labels'],data['root_before'],data['root_after'],data['points'],data['events'])
                tables=dict(canonical_transport_audit=audit)
            elif spec['kind']=='allocation':
                claims,discovery=method(data['before_allocation'],data['after_allocation'],data['points'])
                compose=sys.modules[spec['module']].compose_existing_owners
                candidate,applications=compose(labels,data['after_allocation'],claims)
                tables=dict(allocation_decisions=dict(claims=claims,discovery=discovery,applications=applications))
            else:
                assert spec['kind']=='separation'
                candidate,applications,discovery=method(labels,data['original_labels'],data['before_separation'],data['after_separation'],data['points'])
                tables=dict(ownership_decisions=dict(discovery=discovery,applications=applications.to_dict('records')))
        labels=candidate
        yield spec['stage'],labels,ledger.copy(),tables
