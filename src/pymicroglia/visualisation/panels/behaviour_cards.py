"""Portable state profiles, exact-member traces and frozen original image crops."""
from . import _layout
from ._layout import finish
from . import colour as house_colour
import math
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch
import numpy as np
from ._format import numeric, missing, present, number as display_number
from .behaviour_profiles import state_colour, unpack, number, wrap
from .behaviour_timelines import UNKNOWN, MARKERS, _hatch

def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False

def tidy(axis):
    for name in ['top', 'right']:
        axis.spines[name].set_visible(False)
    axis.tick_params(labelsize=8, width=0.8, length=3)

def profiles(figure, grid, values, settings, axes):
    state = values['state']
    colour = state_colour(state.component)
    nested = grid.subgridspec(1, len(settings['features']), wspace=0.55)
    for index, feature in enumerate(settings['features']):
        axis = figure.add_subplot(nested[0, index])
        axes['profile_' + str(index)] = axis
        row=values['profiles'][index]
        if finite(row['median']):
            axis.scatter([row['median']], [0], color=colour, s=30)
            if finite(row.q25) and finite(row.q75):
                axis.plot([row.q25, row.q75], [0, 0], color=colour, lw=2)
        else:
            axis.text(0.5, 0.5, 'No observed values', transform=axis.transAxes, ha='center', fontsize=8)
        axis.set_yticks([])
        axis.set_ylim(-0.5, 0.5)
        axis.set_title(wrap(feature['measurement'], 23) + '\n' + feature['representation'], fontsize=9)
        axis.set_xlabel((feature['unit'] or 'Recorded units') + '\n' + str(int(row.observed_values)) + ' values; ' + str(int(row.missing_values)) + ' missing', fontsize=8)
        tidy(axis)

def population(figure, grid, values, settings, axes):
    axis = figure.add_subplot(grid)
    axes['population_occupancy'] = axis
    state=values['state']
    labels=[]
    for index,row in enumerate(values['population']):
        axis.scatter(row['x'],row['y'],s=12,alpha=.65,color=state_colour(state.component) if index<2 else house_colour('nan_text'))
        labels.append(row['label'])
    axis.set_yticks(range(3), labels, fontsize=8)
    axis.set_ylim(2.4, -0.4)
    axis.set_xlim(-0.025, 1.025)
    axis.set_xlabel('Saved per-cell fraction; points cover the complete population', fontsize=8)
    tidy(axis)

def ribbon(axis, frame, example, definitions):
    inventory=frame['inventory']
    if inventory.status == 'invalid_clock_order':
        axis.text(0.5, 0.5, 'Nonincreasing clock: state timeline unavailable; traces show points only', transform=axis.transAxes, ha='center', fontsize=8, color=house_colour('circadian_red'))
        axis.set_axis_off()
        return
    for row in frame['intervals']:
        if row['support'] == 'assigned':
            state = definitions[row['state_id']]
            colour = state_colour(state['component'])
            hatch = _hatch(state['component'])
            style = '-'
        elif row['support'] == 'unknown':
            if row['assignment_status'] not in UNKNOWN:
                raise ValueError('State card has an undeclared assignment category')
            _, hatch, colour = UNKNOWN[row['assignment_status']]
            style = '-'
        elif row['support'] == 'unobserved':
            colour, hatch, style = ('white', '', ':')
        else:
            raise ValueError('Unknown saved time support')
        axis.add_patch(Rectangle((row['start_hours'], 0.1), row['width'], 0.8, facecolor=colour, hatch=hatch, edgecolor=house_colour('muted'), lw=0.5, linestyle=style))
    for row in frame['observations']:
        if row['status'] in MARKERS:
            axis.plot([row['hours']], [0.5], marker=MARKERS[row['status']], color=house_colour('slate_tick'), markerfacecolor='white', ms=4, linestyle='none')
    axis.set_xlim(*frame['bounds'])
    axis.axvline(example['hours'], color='black', ls=':', lw=1)
    axis.set_ylim(0, 1)
    axis.set_yticks([])
    axis.tick_params(axis='x', labelsize=7, length=2)
    axis.set_title('Saved state intervals; dotted line marks the exact example', fontsize=8, loc='left')
    for spine in axis.spines.values():
        spine.set_visible(False)

def trace(axis,prepared,feature,example,colour):
    if len(prepared['x']):axis.scatter(prepared['x'],prepared['y'],color=colour,s=10,zorder=3)
    for x,y in prepared['segments']:axis.plot(x,y,color=colour,lw=.75,alpha=.65)
    if len(prepared['selected'][0]):axis.scatter(*prepared['selected'],facecolors='none',edgecolors='black',s=48,lw=1,zorder=4)
    if not len(prepared['x']):axis.text(.5,.5,'No usable original values',transform=axis.transAxes,ha='center',fontsize=8)
    axis.axvline(example['hours'], color='black', ls=':', lw=0.8)
    axis.set_ylabel(wrap(feature['measurement'], 23) + '\n' + feature['representation'] + ' | ' + (feature['unit'] or 'recorded units'), fontsize=8)
    axis.set_xlabel('Original measurement time (h)', fontsize=8)
    axis.ticklabel_format(axis='x', useOffset=False, style='plain')
    tidy(axis)

def imagery(axis, row):
    if row['image_status'] != 'available':
        axis.text(0.5, 0.5, 'Image ' + row['image_status'].replace('_', ' ') + '\n\n' + wrap(row['image_reason'], 37), transform=axis.transAxes, ha='center', va='center', fontsize=9)
        axis.set_axis_off()
        return
    tile = row['tiles'][0]
    pixels,mask=row['pixels'],row['mask']
    axis.imshow(pixels, cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    axis.contour(mask, levels=[0.5], colors=[house_colour('okabe_orange')], linewidths=0.8)
    axis.set_title('Exact recorded cell image\n' + number(tile['hours']) + ' h | source frame ' + str(tile['source_frame_index']), fontsize=9)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)

def draw(values,settings,*,selected_view=None,canvas=None):
    if selected_view not in {None,'profiles','population','traces','images'}:raise ValueError('Unknown state card view')
    definitions={row['state_id']:row for row in settings['state_definitions']}
    count=len(settings['features'])
    profile=selected_view in {None,'profiles'};population_view=selected_view in {None,'population'}
    examples=selected_view in {None,'traces','images'};images=selected_view in {None,'images'}
    heights=([1.35] if profile else [])+([1.35] if population_view else [])
    for _ in settings['example_ids'] if examples else []:heights.extend([.35,3.] if selected_view=='images' else [.35,.45]+[1.1]*count)
    heights=heights or [1.]
    title_extra=.3*(len(settings['title'].splitlines())-1)
    legend_extra=.22*max(0,math.ceil(len(definitions)/4)-1)
    figure=_layout.figure(canvas,figsize=(16,sum(heights)+3.4+title_extra+legend_extra))
    grid=figure.add_gridspec(len(heights),6,height_ratios=heights,left=.16,right=.96,bottom=1.2/figure.get_figheight(),top=1-(1.6+title_extra+legend_extra)/figure.get_figheight(),hspace=1.15,wspace=.55)
    axes={};row_index=0
    if profile:profiles(figure,grid[row_index,:],values,settings,axes);row_index+=1
    if population_view:population(figure,grid[row_index,:],values,settings,axes);row_index+=1
    for example_id in settings['example_ids'] if examples else []:
        frame=values['examples'][example_id];example=frame['example']
        heading=figure.add_subplot(grid[row_index,:]);heading.axis('off');axes['example_heading_'+example_id]=heading
        role='Saved representative' if example['example_role']=='representative' else 'Low-membership assigned example'
        heading.text(0,.75,role+' | '+str(example['movie'])+' | cell '+str(int(example['identity']))+' | '+number(example['hours'])+' h',transform=heading.transAxes,fontsize=10,weight='bold')
        heading.text(0,.05,'Observed bout: '+str(example['bout_status']).replace('_',' ')+'; native membership '+number(example['max_probability'])+' (model conditional, uncalibrated)',transform=heading.transAxes,fontsize=8)
        row_index+=1
        if selected_view=='images':
            axis=figure.add_subplot(grid[row_index,:]);axes['image_'+example_id]=axis;imagery(axis,frame['image']);row_index+=1;continue
        axis=figure.add_subplot(grid[row_index,:4] if images else grid[row_index,:]);axes['ribbon_'+example_id]=axis
        ribbon(axis,frame,example,definitions)
        if images:
            image=figure.add_subplot(grid[row_index:row_index+count+1,4:]);axes['image_'+example_id]=image;imagery(image,frame['image'])
        row_index+=1
        for index,feature in enumerate(settings['features']):
            axis=figure.add_subplot(grid[row_index,:4] if images else grid[row_index,:]);axes['trace_'+example_id+'_'+str(index)]=axis
            trace(axis,frame['traces'][index],feature,example,state_colour(definitions[settings['state_id']]['component']));row_index+=1
    state=values['state'];height=figure.get_figheight()
    figure.suptitle(settings['title'],x=.035,ha='left',y=1-.18/height,fontsize=15,weight='bold')
    figure.text(.035,1-(.58+title_extra)/height,str(int(state.assigned_observations))+' assigned observations; '+str(int(state.cells))+' cells; '+str(int(state.movies))+' recordings; '+str(int(state.confirmed_samples))+' confirmed samples. Profiles: median and observed interquartile range.',fontsize=9)
    handles=[Patch(facecolor=state_colour(row['component']),hatch=_hatch(row['component']),label=settings['state_display_names'].get(row['state_id'],row['label'])) for row in definitions.values()]
    figure.legend(handles=handles,loc='upper left',bbox_to_anchor=(.03,1-(.78+title_extra)/height),ncol=4,fontsize=8,frameon=False)
    if examples and not settings['example_ids']:figure.text(.5,.35,'No assigned observation is available for a real member example',ha='center',fontsize=10)
    note=settings['footnote']+' Unknown ribbon marks: missing measurements (cross), ambiguous (open circle), outside learned distribution (open square); dotted intervals are unobserved.'
    figure.text(.035,.3/height,wrap(note,173),fontsize=8,va='center')
    return figure,axes
