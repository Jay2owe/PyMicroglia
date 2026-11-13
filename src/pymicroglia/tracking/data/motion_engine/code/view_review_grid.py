"""Project adapter for the image-feedback TIFF viewer: readable counts and zoom."""
from pathlib import Path
import argparse,ctypes,importlib.util,json,os,subprocess,sys
import tkinter as tk
from tkinter import ttk
from PIL import Image,ImageTk
import numpy as np

BASE=Path.home()/'.claude/skills/image-feedback/scripts/tiff_viewer.py'
spec=importlib.util.spec_from_file_location('image_feedback_tiff',BASE)
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)

class GridViewer(base.Viewer):
    def __init__(self,path):
        if os.name=='nt':
            try:ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except (AttributeError,OSError):pass
        self.manifest=json.loads((path.parent/'grid_manifest.json').read_text())
        self.counts=np.asarray(self.manifest['per_movie_per_frame_changed_pixels'])
        self.mode='all'
        super().__init__(path)
        self.root.title(f"{self.manifest['attempt']} - Before | After | Changes")
        self.root.geometry('1450x950')
        self.image_label.pack_forget()
        body=ttk.Frame(self.root);body.pack(fill='both',expand=True,before=self.slice_label.master)
        side=ttk.Frame(body,padding=8);side.pack(side='right',fill='y')
        self.summary=ttk.Label(side,font=('Arial',12,'bold'));self.summary.pack(anchor='w',pady=8)
        ttk.Label(side,text='Changed pixels\nMovie          This frame / Whole movie',font=('Arial',11)).pack(anchor='w')
        self.rows=[]
        for entry in self.manifest['entries']:
            label=ttk.Label(side,font=('Consolas',12));label.pack(anchor='w',pady=5);self.rows.append(label)
        ttk.Label(side,text='Red: exact edits\nYellow boxes: locators only',font=('Arial',11)).pack(anchor='w',pady=12)
        for title,command in [('Previous changed frame',lambda:self.next_change(-1)),('Next changed frame',lambda:self.next_change(1)),('First frame',lambda:self.show_index(0)),('Fit all movies',lambda:self.zoom('all')),('Fit width (scroll vertically)',lambda:self.zoom('width')),('100% (native pixels)',lambda:self.zoom('native'))]:
            ttk.Button(side,text=title,command=command).pack(fill='x',pady=3)
        field=ttk.Frame(body);field.pack(side='left',fill='both',expand=True)
        self.canvas=tk.Canvas(field,bg='#171717',highlightthickness=0)
        vs=ttk.Scrollbar(field,orient='vertical',command=self.canvas.yview)
        hs=ttk.Scrollbar(field,orient='horizontal',command=self.canvas.xview)
        self.canvas.configure(yscrollcommand=vs.set,xscrollcommand=hs.set)
        vs.pack(side='right',fill='y');hs.pack(side='bottom',fill='x');self.canvas.pack(fill='both',expand=True)
        self.canvas.bind('<Configure>',lambda event:self.render())
        self.canvas.bind('<Shift-MouseWheel>',lambda event:(self.canvas.yview_scroll(-int(event.delta/120),'units'),'break')[-1])
        self.status.configure(text='Arrows/wheel: frames | Shift+wheel: scroll')
        self.show_index(int(np.argmax(self.counts.sum(axis=0))))

    def zoom(self,mode):
        self.mode=mode;self.render()

    def next_change(self,direction):
        candidates=np.flatnonzero(self.counts.sum(axis=0)>0)
        if not len(candidates):return
        eligible=candidates[candidates>self.index] if direction>0 else candidates[candidates<self.index]
        self.show_index(int(eligible[0] if direction>0 and len(eligible) else eligible[-1] if len(eligible) else candidates[0] if direction>0 else candidates[-1]))

    def render(self):
        if not hasattr(self,'canvas'):return
        current=self.counts[:,self.index]
        self.summary.configure(text=f"Frame {self.index+1}/{self.count}\n{int(current.sum()):,} pixels this frame\n{int(self.counts.sum()):,} across all movies")
        for label,entry,n in zip(self.rows,self.manifest['entries'],current):
            label.configure(text=f"{entry['well']:8} {int(n):5,} / {entry['changed_pixels']:,}")
        image=self.cache.get(self.index)
        if image is None or self.closed:return
        width=max(1,self.canvas.winfo_width()-16);height=max(1,self.canvas.winfo_height()-16)
        scale=min(width/image.width,height/image.height) if self.mode=='all' else width/image.width if self.mode=='width' else 1
        shown=image.resize((max(1,round(image.width*scale)),max(1,round(image.height*scale))),Image.Resampling.NEAREST)
        self.photo=ImageTk.PhotoImage(shown)
        self.canvas.delete('all');self.canvas.create_image(0,0,anchor='nw',image=self.photo)
        self.canvas.configure(scrollregion=(0,0,shown.width,shown.height))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=['open','foreground','probe']);p.add_argument('path',type=Path);a=p.parse_args();path=a.path.resolve()
    info=base.stack_info(path)
    if a.command=='foreground':GridViewer(path).run()
    elif a.command=='probe':print(json.dumps(info))
    else:
        exe=Path(sys.executable).with_name('pythonw.exe')
        proc=subprocess.Popen([str(exe if exe.exists() else sys.executable),str(Path(__file__).resolve()),'foreground',str(path)],creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name=='nt' else 0)
        print(json.dumps({**info,'pid':proc.pid}))
