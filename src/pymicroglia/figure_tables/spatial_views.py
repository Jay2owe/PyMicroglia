"""Split prepared spatial encodings into independently selectable views."""
from .spatial import prepare
from .prepared import PreparedViews


def _prepare(source,options,kind):
    table,auxiliary,cfg,subtitle,footnote = prepare(source,options,kind)
    def data(rows):
        return dict(table=rows,cfg=cfg,subtitle=subtitle,footnote=footnote)
    if kind in {'expansion','gaps'}:
        result = {'field':data(table.loc[table.record.isin(['pixel','arrow'])])}
        if kind=='gaps':
            result['coverage']=data(table.loc[table.record.eq('coverage')])
    elif kind=='period':
        result={key:data(table.loc[table.tile.eq('period') if key=='period' else ~table.tile.eq('period')])
                for key in ('period','phase')}
    elif kind=='progression':
        result={'positions':data(table.loc[table.record.eq('cell')]),'matrix':data(table)}
    elif kind=='neighbours':
        result={'connections':data(table.loc[table.record.isin(['pair','cell'])]),'null':data(table.loc[table.record.eq('null_bin')])}
    else:
        result={'timing':data(table)}
    return PreparedViews(result, auxiliary=auxiliary,
                         wording=dict(subtitle=subtitle,footnote=footnote)),auxiliary.get('statistics.csv')


def expansion(source,options):
    return _prepare(source,options,'expansion')


def period(source,options):
    return _prepare(source,options,'period')


def progression(source,options):
    return _prepare(source,options,'progression')


def timing(source,options):
    return _prepare(source,options,'timing')


def neighbours(source,options):
    return _prepare(source,options,'neighbours')


def gaps(source,options):
    return _prepare(source,options,'gaps')
