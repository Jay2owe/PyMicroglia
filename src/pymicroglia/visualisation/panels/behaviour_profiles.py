"""Portable drawing of frozen criteria and observed state profiles only."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
import ast
import colorsys
import json
import math
from pathlib import Path
import textwrap
import matplotlib.pyplot as plt
import numpy as np
from ._format import numeric, missing, present, number as display_number
COLOURS = [house_colour('blue'), house_colour('circadian_red'), house_colour('circadian_teal'), house_colour('circadian_amber'), house_colour('circadian_purple'), house_colour('grade_green')]
STATUS_COLOURS = {'pass': house_colour('circadian_teal'), 'accepted': house_colour('circadian_teal'), 'supported': house_colour('circadian_teal'), 'completed': house_colour('circadian_teal'), 'fail': house_colour('circadian_red'), 'failed': house_colour('circadian_red'), 'unavailable': house_colour('circadian_amber'), 'inconclusive': house_colour('circadian_amber'), 'no_supported_states': house_colour('nan_text')}

def state_colour(component):
    """Stable model-component colours; extending the vocabulary never cycles six."""
    component = int(component)
    if component < len(COLOURS):
        return COLOURS[component]
    return colorsys.hsv_to_rgb(component * 0.6180339887498949 % 1.0, 0.55 + 0.12 * (component % 2), 0.72 + 0.12 * (component // 2 % 2))

def unpack(value):
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            try:
                return ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return value
    return value

def number(value):
    try:
        return f'{float(value):.3g}' if math.isfinite(float(value)) else 'unavailable'
    except (TypeError, ValueError):
        return 'unavailable'

def wrap(value, width):
    return '\n'.join((textwrap.fill(line, width) for line in str(value).splitlines()))

def criterion(value, ordinal):
    labels = {'independent_groups': 'Independent validation groups', 'multiple_components': 'More than one candidate component', 'independent_multimodality': 'Corrected independent multimodality evidence', 'assignment_coverage': 'Fraction receiving an assignment', 'density_gain_over_single_component': 'Predictive density gain over one component'}
    if value.startswith('component_group_count:'):
        return 'Component ' + str(int(value.split(':')[1]) + 1) + ': independent groups'
    if value.startswith('component_separation:'):
        return 'Component ' + str(int(value.split(':')[1]) + 1) + ': silhouette separation'
    if value.startswith('grouped_refit:'):
        return 'Repeat ' + str(ordinal) + ': one learning group withheld'
    return labels.get(value, value.replace('_', ' '))

def observed(value, name):
    value = unpack(value)
    if isinstance(value, dict):
        if 'adjusted_rand' in value:
            recalls = [row['recall'] for row in value.get('mapping', [])]
            return 'Label agreement ' + number(value['adjusted_rand']) + '; smallest matched fraction ' + (number(min(recalls)) if recalls else 'unavailable')
        if 'minimum_adjusted_rand' in value:
            return 'Agreement at least ' + number(value['minimum_adjusted_rand']) + '; matched fraction at least ' + number(value['minimum_component_recall']) + '; coverage at least ' + number(value['minimum_assignment_coverage'])
        return str(value)
    if isinstance(value, (list, tuple)):
        return ', '.join(('table endpoint' if name == 'independent_multimodality' and item == 0 else number(item) for item in value))
    if value is None or (not isinstance(value, str) and missing(value)):
        return 'not applicable'
    return value if isinstance(value, str) else number(value)

def draw(prepared,settings,*,selected_view=None,canvas=None):
    meta=settings['metadata'];support=settings['view']=='support'
    allowed=('criteria','outcomes') if support else ('values','membership')
    if selected_view is not None and selected_view not in allowed:raise ValueError('Unknown state profile view')
    if support:
        rows=prepared['rows'];total=prepared['total'];height=max(7.,total+3.5)
        figure,axis=_layout.subplots(canvas,figsize=(14,height))
        figure.subplots_adjust(left=.04,right=.97,top=1-1.65/height,bottom=1.35/height)
        axis.set_xlim(0,1);axis.set_ylim(0,total+.55);axis.axis('off')
        headings=[(.01,'Saved check')]
        if selected_view!='outcomes':headings.extend([(.35,'Observed result'),(.6,'Declared requirement')])
        if selected_view!='criteria':headings.append((.35 if selected_view=='outcomes' else .86,'Outcome'))
        for x,label in headings:axis.text(x,total+.35,label,weight='bold',fontsize=9,va='center')
        cursor=total
        for row in rows:
            y=cursor-row['height']/2;color=STATUS_COLOURS.get(row['status'],house_colour('nan_text'))
            axis.axhspan(cursor-row['height'],cursor,color=color,alpha=.055,linewidth=0)
            axis.plot([0,1],[cursor-row['height'],cursor-row['height']],color=house_colour('actogram_dark'),lw=.5)
            texts=[(.01,row['label'])]
            if selected_view!='outcomes':texts.extend([(.35,row['measured']),(.6,row['required'])])
            for x,label in texts:axis.text(x,y,label,fontsize=8.5,va='center')
            if selected_view!='criteria':axis.text(.35 if selected_view=='outcomes' else .86,y,wrap(row['status'].replace('_',' '),18),fontsize=8,color=color,va='center',weight='bold')
            cursor-=row['height']
        axis.set_title(settings['group']+' | page '+str(settings['part']),loc='left',fontsize=11,pad=16)
        axes={selected_view or 'criteria':axis}
    else:
        features=prepared['features'];nstates=len(settings['states'])
        height=max(6.,3.+len(features)*max(1.65,.38*nstates+1.))
        figure,array=_layout.subplots(canvas,len(features),1,figsize=(13,height),squeeze=False)
        figure.subplots_adjust(left=.22,right=.68,top=1-1.25/height,bottom=1.2/height,hspace=1.)
        axes={}
        for index,feature in enumerate(features):
            axis=array[index,0];axes['feature_'+str(index)]=axis
            for position,row in enumerate(feature['rows']):
                color=state_colour(row['component'])
                if selected_view!='membership':
                    if row['valid']:
                        axis.plot([row['lower'],row['upper']],[position,position],color=color,lw=2.5)
                        axis.scatter([row['median']],[position],color=color,s=38,zorder=3)
                    else:axis.text(.02,position,'No observed values',transform=axis.get_yaxis_transform(),color=house_colour('nan_text'),fontsize=8)
                if selected_view!='values':
                    axis.text(.02 if selected_view=='membership' else 1.04,position,row['counts'],transform=axis.get_yaxis_transform(),fontsize=8,va='center')
            axis.set_yticks(range(nstates),[wrap(row['label'],25) for row in feature['rows']])
            axis.set_ylim(nstates-.5,-.5)
            if selected_view=='membership':axis.set_xticks([])
            else:axis.set_xlabel(feature['measurement']+' ('+feature['unit']+')',fontsize=9)
            axis.set_title(feature['measurement']+' | contributing observations' if selected_view=='membership' else feature['representation'].capitalize()+' observed measurements',fontsize=10,loc='left')
            axis.tick_params(labelsize=9);finish(axis)
    figure.suptitle(settings['title'],fontsize=15,weight='bold',x=.04,ha='left',y=1-.25/height)
    figure.text(.04,1-.82/height,'Saved decision: '+str(meta['status']).replace('_',' '),fontsize=11,color=STATUS_COLOURS.get(meta['status'],house_colour('slate_text')),weight='bold')
    if not support and prepared['assignment_note']:figure.text(.4,1-.82/height,prepared['assignment_note'],fontsize=9)
    if support:figure.text(.04,.83/height,wrap(meta['reason'],145),fontsize=8.5,va='top')
    figure.text(.04,.39/height,wrap(settings['footnote'],148),fontsize=8,va='center')
    return figure,axes
