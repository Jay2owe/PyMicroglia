"""Place resolved wording without covering the plotted evidence."""
import textwrap


def finish(figure, wording):
    width,height = figure.get_size_inches()
    wrap = max(30, int(width*7))
    title = textwrap.fill(wording.title, wrap)
    subtitle = textwrap.fill(wording.subtitle, wrap, replace_whitespace=False)
    footer = '\n'.join(textwrap.fill(value,wrap,replace_whitespace=False)
                       for value in (wording.footnote,wording.note) if value)
    top_lines = len((title+'\n'+subtitle).strip().splitlines())
    bottom_lines = len(footer.splitlines())
    top = min(.4,(.3+.32*top_lines)/height)
    bottom = min(.45,(.3+.25*bottom_lines)/height) if footer else .03
    if title:
        figure.text(.5,.99,title,ha='center',va='top',fontsize=22)
    if subtitle:
        figure.text(.5,1-(.35+.35*len(title.splitlines()))/height,subtitle,
                    ha='center',va='top',fontsize=15)
    if footer:
        figure.text(.02,.12/height,footer,ha='left',va='bottom',fontsize=12)
    for axis in figure.axes:
        for getter,setter,limit in ((axis.get_xlabel,axis.set_xlabel,38),(axis.get_ylabel,axis.set_ylabel,24)):
            setter('\n'.join(textwrap.fill(line,limit) for line in getter().splitlines()))
    figure.set_layout_engine('constrained',rect=(.01,bottom,.98,1-top-bottom))
