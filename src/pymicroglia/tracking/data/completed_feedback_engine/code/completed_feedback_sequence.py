"""Joined accepted operations with fresh dependency arrays and scientific tables.

Callers supply native measurements/proposals generated for the run being tested.
Movie keys route independent recordings, never correction eligibility. A callback
can record or compare every boundary before the next operation executes.
"""
import initial_completed_feedback_composition as initial
import fragment_contact_composition as fragment
import owner_episode_composition as owner
import ownerless_contact_composition as ownerless
import embedded_handoff_composition as embedded
import review_union_composition as union
import optical_source_composition as optical
import recurrent_branch_composition as branch
import process_group_composition_v2 as process
import repeated_name_composition as repeated
import duplicate_contact_composition as duplicate
import terminal_owner_composition as terminal
import handoff_contact_composition as handoff
import dim_exchange_composition as dim
import anchored_memory_composition as anchored
import shared_birth_composition as birth
import detector_gap_composition as detector
import recovery_sequence_composition as recovery
import current_path_sequence_composition as current
import contact_sequence_composition as contact
import reviewed_ownership_composition as reviewed
import final_ownership_composition as final


def apply(movies,density_parameters,operations,source_frame_offset,
          revise_isolated_prior_reassignment,callback,inventory_callback):
    """Run the same accepted sequence for every movie and return final arrays.

Each input movie has labels, ledger, raw, points, birth_labels, births,
presence_input, presence_saved, presence_parameters, recovery_history,
physical, physical_evidence and final_evidence. Native birth/physical and final
evidence must already use their declared original temporal alignments.
"""
    states={};discovery_movies={}
    for key,data in movies.items():
        labels,ledger=data['labels'].copy(),data['ledger'].copy()
        raw,points=data['raw'],data['points']
        native_baseline=labels.copy()
        labels,tables,statistics=initial.apply(labels,points,density_parameters)
        callback(key,'initial','lineage_union',labels,ledger,tables)
        restored=labels.copy()
        tables={name:fragment.table_boundary(table) for name,table in tables.items()}
        labels,tables,protected=fragment.fragments(labels,points,tables)
        callback(key,'fragment','fragments',labels,ledger,tables)
        fragment_audit=fragment.table_boundary(tables['fragment_reunion_audit'])
        labels,tables,protected=fragment.contacts(labels,raw,points,protected)
        callback(key,'fragment','contacts',labels,ledger,tables)
        labels,tables,protected=fragment.remnants(labels,tables,protected)
        callback(key,'fragment','remnants',labels,ledger,tables)
        for stage,labels,tables,protected in owner.apply(labels,raw,points,protected):
            callback(key,'owner',stage,labels,ledger,tables)
        lineage_labels=labels.copy()
        del protected
        for stage,labels,ledger,tables in ownerless.apply(labels,ledger,raw,points,restored):
            callback(key,'ownerless',stage,labels,ledger,tables)
        for stage,labels,ledger,tables in embedded.apply(labels,ledger,raw,points,restored):
            callback(key,'embedded',stage,labels,ledger,tables)
            if stage=='coupled_exchange':embedded_labels=labels.copy()
            if stage=='terminal_handoff':interface_labels=labels.copy()
        for stage,labels,ledger,tables in union.apply(labels,ledger,raw,points,restored):
            callback(key,'union',stage,labels,ledger,tables)
            if stage=='reviewed_union':union_labels=labels.copy()
        for stage,labels,ledger,tables in optical.apply(labels,ledger,raw,points,restored):
            callback(key,'optical',stage,labels,ledger,tables)
        pre_branch_labels=labels.copy()
        for stage,labels,ledger,tables in branch.apply(labels,ledger,raw,points,restored):
            callback(key,'branch',stage,labels,ledger,tables)
        branch_applications=fragment.table_boundary(tables['recurrent_branch_applications'])
        discovery_movies[key]=dict(points=points,raw=raw,fragment_audit=fragment_audit,
            lineage_labels=lineage_labels,embedded_labels=embedded_labels,restored=restored,
            interface_labels=interface_labels,union_labels=union_labels,branch_labels=labels,
            pre_branch_labels=pre_branch_labels,ledger=ledger,branch_applications=branch_applications)
        states[key]=dict(labels=labels,ledger=ledger,restored=restored,native_baseline=native_baseline)
        # Discovery owns these snapshots until the process inventory and its
        # applications finish. Do not also retain the last movie in locals.
        del lineage_labels,embedded_labels,interface_labels,union_labels,pre_branch_labels
        del fragment_audit,branch_applications,tables,statistics

    for stage,table in process.inventory(discovery_movies):
        inventory_callback(stage,table)
    neighbours=process.csv_boundary(table)
    results={}
    for key,data in movies.items():
        state=states.pop(key);restored=state['restored'];raw,points=data['raw'],data['points']
        for stage,labels,ledger,tables in process.apply(discovery_movies[key],neighbours[neighbours.well.eq(key)].drop(columns=['well'])):
            callback(key,'process',stage,labels,ledger,tables)
        # All later stages use the new labels plus restored/native ancestors;
        # none reads the earlier discovery snapshots or branch output.
        del discovery_movies[key]
        state.pop('labels');state.pop('ledger')
        for stage,labels,ledger,tables in repeated.apply(labels,ledger,raw,points,restored):
            callback(key,'repeated',stage,labels,ledger,tables)
        before_allocation=labels.copy()
        for stage,labels,ledger,tables in duplicate.apply(labels,ledger,raw,points,restored):
            callback(key,'duplicate',stage,labels,ledger,tables)
            if stage=='duplicate_sites':after_allocation=labels.copy()
        for stage,labels,ledger,tables in terminal.apply(labels,ledger,raw,points,restored):
            callback(key,'terminal',stage,labels,ledger,tables)
        for stage,labels,ledger,tables in handoff.apply(labels,ledger,raw,points,restored):
            callback(key,'handoff',stage,labels,ledger,tables)
            if stage=='raw_rejoin':before_separation=labels.copy()
            if stage=='brief_contact':after_separation=labels.copy()
        for family,block in [('dim',dim),('anchored',anchored)]:
            for stage,labels,ledger,tables in block.apply(labels,ledger,raw,points,restored):
                callback(key,family,stage,labels,ledger,tables)
        for stage,labels,ledger,tables in birth.apply(labels,ledger,raw,points,restored,
                data['birth_labels'],data['births'],source_frame_offset,revise_isolated_prior_reassignment):
            callback(key,'birth',stage,labels,ledger,tables)
        for stage,labels,ledger,tables in detector.apply(labels,ledger,raw,points,restored,source_frame_offset):
            callback(key,'detector',stage,labels,ledger,tables)
        for family,block in [('recovery',recovery),('current',current),('contact',contact)]:
            for stage,labels,ledger,tables,arrays in block.apply(labels,ledger,raw,points,restored,
                    data['presence_input'],data['presence_saved'],data['presence_parameters'],data['recovery_history'],operations[family]):
                callback(key,family,stage,labels,ledger,tables,arrays=arrays)
        for stage,labels,ledger,tables in reviewed.apply(labels,ledger,raw,points,restored,
                data['presence_input'],data['presence_saved'],data['presence_parameters'],data['recovery_history'],
                data['physical'],data['physical_evidence'],operations['reviewed']):
            callback(key,'reviewed',stage,labels,ledger,tables)
        evidence={stage:dict(values) for stage,values in data['final_evidence'].items()}
        for spec in operations['final']:
            values=evidence[spec['stage']]
            if spec['kind'] in ['gap','restoration']:values['original_labels']=restored
            elif spec['kind']=='allocation':
                values.update(before_allocation=before_allocation,after_allocation=after_allocation)
            else:
                values.update(original_labels=state['native_baseline'],before_separation=before_separation,after_separation=after_separation)
        for stage,labels,ledger,tables in final.apply(labels,ledger,evidence,operations['final'],
                source_frame_offset=source_frame_offset):
            callback(key,'final',stage,labels,ledger,tables)
        results[key]=(labels,ledger)
    return results
