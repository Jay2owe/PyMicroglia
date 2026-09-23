"""Place the original complete diagnostic grid on a house-style canvas."""
from . import colour
from . import rhythm_audit


def grid(figure,data,*,style,**options):
    setup=data['config'];resolved=setup['resolved'];estimators=setup['estimators'];detrends=setup['detrends']
    width,height=setup['nominal_width'],setup['nominal_height']
    figure.set_size_inches(width,height)
    config={**setup,'figsize':[width,height],'rect':[.06,1.75/height,.925,1-3.75/height],
        'shared_comparisons':True,'descriptive_cosinor':bool(resolved['params'].get('descriptive_cosinor',False)),
        'comparison_row_height':max(.82,.16*(len(estimators)+2)),
        'detrend_labels':setup['detrend_labels'],'estimator_labels':setup['estimator_labels'],
        'preprocessors':['raw','median'],'preprocessor_labels':{'raw':'Raw input','median':f"Median ({setup['median_points']} sampling intervals)"},
        'period_min_hours':resolved['period_min_hours'],'period_max_hours':resolved['period_max_hours'],'min_cycles':resolved['min_cycles'],
        'title':'','subtitle':'','footnote':'','header_x':.06,'title_y':1-.24/height,'subtitle_y':1-.78/height,'legend_y':1-.75/height,'footnote_y':.18/height,
        'colours':{key:colour(value) for key,value in {'raw':'dark','median':'teal','reference':'muted','warning':'orange','ink':'dark','caption':'slate_text','page':'white'}.items()},
        'font':{'title':22,'subtitle':14,'legend':11,'column':11,'row':11,'annotation':9,'small':9,'tick':9,'axis':11,'footnote':10},
        'stroke':{'axis':.8,'tick_length':3,'guide':.6,'data':.6,'line':1.,'fit':1.5},'point_area':9,'component_marker':4}
    return rhythm_audit.draw(figure,tuple(config['rect']),data['table'],data['results'],data['components'],data['spectra'],config)
