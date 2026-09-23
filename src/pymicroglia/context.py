"""Read-only package guidance, using only the Python standard library."""
from importlib.resources import files
import re

TOPICS = ('overview','quickstart','inputs','tracking','measure','rhythms',
          'states','clustering','lifecycle','figures','pipelines','records','troubleshooting')


def topics():
    return TOPICS


def read(topic=None, *, format='text'):
    """Read a named guide, or the overview; JSON format returns a plain mapping."""
    topic = topic or 'overview'
    if topic not in TOPICS:
        raise ValueError(f'Unknown guide {topic!r}; choose {", ".join(TOPICS)}')
    if format not in {'text','json'}:
        raise ValueError('format must be text or json')
    content=files('pymicroglia').joinpath('guide',topic+'.md').read_text(encoding='utf-8')
    if format=='text':return content
    return {'topic':topic,'title':content.splitlines()[0].lstrip('# '),'content':content}


def search(query, *, limit=5):
    """Rank guide topics by query-word matches; never open a recording or run science."""
    if isinstance(limit,bool) or not isinstance(limit,int) or limit<1:
        raise ValueError('limit must be a positive integer')
    words=set(re.findall(r'[\w-]+',str(query).lower()))
    if not words:return []
    matches=[]
    for topic in TOPICS:
        row=read(topic,format='json');body=row['content'].lower()
        score=sum(body.count(word) for word in words)
        if score:
            paragraphs=row['content'].split('\n\n')
            excerpt=max(paragraphs,key=lambda text:sum(text.lower().count(word) for word in words))
            matches.append({'topic':topic,'title':row['title'],'score':score,'excerpt':excerpt})
    return sorted(matches,key=lambda row:(-row['score'],row['topic']))[:limit]
