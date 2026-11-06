from pathlib import Path
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
OUT=Path(__file__).resolve().parents[1]; DATA=OUT/'data'; FIG=OUT/'figures'
FONT='/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc'
font_manager.fontManager.addfont(FONT)
plt.rcParams.update({'font.family':font_manager.FontProperties(fname=FONT).get_name(),'font.size':14,'axes.unicode_minus':False,'svg.fonttype':'path','savefig.facecolor':'white','axes.spines.top':False,'axes.spines.right':False,'axes.labelcolor':'#183047','text.color':'#183047','axes.titleweight':'bold'})
BLUE='#2454A4';TEAL='#008B83';INK='#183047';GRAY='#64748B';RED='#B44949'; LIGHT='#EFF4F8'
def load(n): return json.loads((DATA/(n+'.json')).read_text())
def canvas(title,sub=''):
 f=plt.figure(figsize=(16,9),layout=None)
 f.text(.045,.938,title,fontsize=25,weight='bold',color=INK)
 if sub:f.text(.045,.889,sub,fontsize=13,color=GRAY)
 return f
FIG.mkdir(exist_ok=True)
def finish(f,name,foot):
 f.text(.045,.035,foot,fontsize=11,color=GRAY,va='bottom')
 for ext in ['png','svg']:
  f.savefig(FIG/(name+'.'+ext),dpi=240)
 plt.close(f)
def table(title,sub,headers,rows,name,foot,widths=None,size=15,top=.84,bottom=.12):
 f=canvas(title,sub);ax=f.add_axes([.045,bottom,.91,top-bottom]);ax.axis('off')
 t=ax.table(cellText=rows,colLabels=headers,colWidths=widths,cellLoc='left',colLoc='left',bbox=[0,0,1,1]);t.auto_set_font_size(False);t.set_fontsize(size)
 for (r,c),cell in t.get_celld().items():
  cell.PAD=.06;cell.set_edgecolor('white');cell.set_linewidth(1.8)
  if r==0:cell.set_facecolor(INK);cell.get_text().set_color('white');cell.get_text().set_weight('bold')
  else:
   cell.set_facecolor('white' if r%2==0 else LIGHT)
   if c==1 and len(headers)==3:cell.get_text().set_color(BLUE)
   if c==2 and len(headers)==3:cell.get_text().set_color(TEAL)
 finish(f,name,foot)
def pct(v):return '—' if v is None else f'{100*v:.1f}%'
def num(v,n=3):return '—' if v is None else f'{v:.{n}f}'
def trip(v):return '/'.join(f'{x:.2f}' for x in v)
