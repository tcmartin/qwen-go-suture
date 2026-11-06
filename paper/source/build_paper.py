"""Render one canonical manuscript to an editable standalone TeX file and review PDF.
The PDF uses ReportLab; it is not represented as a LaTeX compilation.
No model execution or network access.
"""
from pathlib import Path
import html,re,json,hashlib
import manuscript as m
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,KeepTogether,PageBreak
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
ROOT=Path(__file__).resolve().parents[1]
PDF=ROOT/'output/pdf/go-hidden-roundtrip-preprint.pdf';PDF.parent.mkdir(parents=True,exist_ok=True)
W,H=A4;M=53;WIDTH=W-2*M
styles={
 'title':ParagraphStyle('title',fontName='Times-Bold',fontSize=20,leading=23,alignment=TA_CENTER,spaceAfter=10),
 'subtitle':ParagraphStyle('subtitle',fontName='Times-Italic',fontSize=12,leading=15,alignment=TA_CENTER,spaceAfter=10),
 'meta':ParagraphStyle('meta',fontName='Helvetica',fontSize=8.8,leading=12,alignment=TA_CENTER,spaceAfter=7),
 'h':ParagraphStyle('h',fontName='Times-Bold',fontSize=13.5,leading=16,spaceBefore=13,spaceAfter=7,keepWithNext=True),
 'subh':ParagraphStyle('subh',fontName='Times-Bold',fontSize=11.7,leading=14,spaceBefore=10,spaceAfter=6,keepWithNext=True),
 'p':ParagraphStyle('p',fontName='Times-Roman',fontSize=10.5,leading=13.7,spaceAfter=7,allowWidows=0,allowOrphans=0),
 'caption':ParagraphStyle('caption',fontName='Times-Italic',fontSize=9.4,leading=12,spaceBefore=7,spaceAfter=6,keepWithNext=True),
 'cell':ParagraphStyle('cell',fontName='Times-Roman',fontSize=9.2,leading=11.5),
 'th':ParagraphStyle('th',fontName='Times-Bold',fontSize=9.2,leading=11.5),
 'hash':ParagraphStyle('hash',fontName='Courier',fontSize=7.4,leading=10,splitLongWords=True),
 'eq':ParagraphStyle('eq',fontName='Times-Italic',fontSize=10,leading=13,spaceBefore=3,spaceAfter=9,leftIndent=12,rightIndent=12),
 'ref':ParagraphStyle('ref',fontName='Times-Roman',fontSize=9,leading=11,spaceAfter=4),
}
def P(s,style='p'):return Paragraph(html.escape(str(s)),styles[style])
def table_widths(n,caption):
 if 'Table A1' in caption:return [43,WIDTH-43]
 if 'Table B1' in caption:return [155,WIDTH-155]
 if n==3:return [WIDTH*.53,WIDTH*.235,WIDTH*.235]
 if n==4:return [WIDTH*.41,WIDTH*.17,WIDTH*.24,WIDTH*.18]
 if n==5:return [WIDTH*.32,WIDTH*.17,WIDTH*.18,WIDTH*.18,WIDTH*.15]
 return [WIDTH/n]*n

def table(caption,head,rows):
 widths=table_widths(len(head),caption)
 body=[[P(x,'th') for x in head]]
 for row in rows:
  body.append([P(x,'hash' if re.fullmatch(r'[a-f0-9]{40,64}',str(x)) else 'cell') for x in row])
 t=Table(body,colWidths=widths,repeatRows=1,hAlign='LEFT')
 t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5),('LINEABOVE',(0,0),(-1,0),.8,colors.HexColor('#26394b')),('LINEBELOW',(0,0),(-1,0),.5,colors.HexColor('#26394b')),('LINEBELOW',(0,-1),(-1,-1),.6,colors.HexColor('#26394b')),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f6f8')])]))
 return [P(caption,'caption'),t,Spacer(1,9)]

story=[P(m.TITLE,'title'),P(m.SUBTITLE,'subtitle'),P(m.AUTHOR,'meta'),P(m.DATE+' | Research preprint; not peer reviewed','meta'),Spacer(1,10)]
for b in m.blocks:
 if b[0]=='h':story.append(P(b[1],'subh' if re.match(r'\d+\.\d+',b[1]) else 'h'))
 elif b[0]=='p':story.append(P(b[1]))
 elif b[0]=='eq':story.append(P(b[1],'eq'))
 elif b[0]=='table':story.extend(table(*b[1:]))
story.append(P('References','h'))
for i,(ref,url) in enumerate(m.refs,1):
 story.append(Paragraph(f'[{i}] {html.escape(ref)} <link href="{html.escape(url)}" color="#225b80">{html.escape(url)}</link>',styles['ref']))

def footer(c,doc):
 c.saveState();c.setStrokeColor(colors.HexColor('#c7ced5'));c.setLineWidth(.4);c.line(M,39,W-M,39)
 c.setFont('Helvetica',7.5);c.setFillColor(colors.HexColor('#56616c'));c.drawString(M,27,'Go hidden-state round trip | Evidence through stage 029 | Research preprint');c.drawRightString(W-M,27,str(doc.page));c.restoreState()
doc=SimpleDocTemplate(str(PDF),pagesize=A4,rightMargin=M,leftMargin=M,topMargin=48,bottomMargin=52,title=m.TITLE,author=m.AUTHOR,pageCompression=1)
doc.build(story,onFirstPage=footer,onLaterPages=footer)

# One self-contained source file. Embedded bibliography avoids a required BibTeX pass.
def esc(s):
 mapping={'\\':r'\textbackslash{}','&':r'\&','%':r'\%','$':r'\$','#':r'\#','_':r'\_','{':r'\{','}':r'\}','~':r'\textasciitilde{}','^':r'\textasciicircum{}'}
 s=''.join(mapping.get(c,c) for c in str(s))
 s=re.sub(r'(?<![A-Za-z0-9])([a-f0-9]{40,64})(?![A-Za-z0-9])',lambda z:r'\texttt{'+r'\allowbreak '.join(z[0][i:i+8] for i in range(0,len(z[0]),8))+'}',s)
 return s
tex=[r'''\documentclass[11pt]{article}
\usepackage[a4paper,margin=22mm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{amsmath,amssymb,booktabs,longtable,array}
\usepackage[hidelinks,breaklinks]{hyperref}
\usepackage{microtype}
\setlength{\parindent}{0pt}
\setlength{\parskip}{6pt}
\setlength{\emergencystretch}{3em}
\newcommand{\draftnotice}{Research preprint; not peer reviewed}
\begin{document}''',r'\title{'+esc(m.TITLE)+r'\\[6pt]\large '+esc(m.SUBTITLE)+'}',r'\author{'+esc(m.AUTHOR)+'}',r'\date{'+esc(m.DATE)+r'\\\small\draftnotice}',r'\maketitle']
for b in m.blocks:
 if b[0]=='h':tex.append(('\\subsection*{' if re.match(r'\d+\.\d+',b[1]) else '\\section*{')+esc(b[1])+'}')
 elif b[0]=='p':tex.append(esc(b[1])+'\n')
 elif b[0]=='eq':
  eq=b[2]
  if len(eq)>190:
   if r',\quad \widehat F' in eq:eq=eq.replace(r',\quad \widehat F',r',\\ \widehat F')
   if r',\qquad \Delta' in eq:eq=eq.replace(r',\qquad \Delta',r',\\ \Delta')
   tex.append(r'\[\begin{gathered}'+eq+r'\end{gathered}\]')
  else:tex.append(r'\['+eq+r'\]')
 elif b[0]=='table':
  cap,head,rows=b[1:];n=len(head)
  # column widths include inter-column padding and fit \linewidth.
  weights=[w/WIDTH for w in table_widths(n,cap)];cols=''.join(r'>{\raggedright\arraybackslash}p{'+str(round(.9*x,4))+r'\linewidth}' for x in weights)
  tex.extend([r'\begingroup\small\setlength{\tabcolsep}{3pt}',r'\begin{longtable}{'+cols+'}',r'\multicolumn{'+str(n)+r'}{p{.96\linewidth}}{\textit{'+esc(cap)+r'}}\\[5pt]',r'\toprule',' & '.join(r'\textbf{'+esc(x)+'}' for x in head)+r'\\\midrule',r'\endfirsthead',r'\toprule',' & '.join(r'\textbf{'+esc(x)+'}' for x in head)+r'\\\midrule',r'\endhead'])
  tex.extend(' & '.join(esc(x) for x in row)+r'\\[3pt]' for row in rows)
  tex.extend([r'\bottomrule\end{longtable}\endgroup'])
tex.append(r'\begin{thebibliography}{9}')
for i,(ref,url) in enumerate(m.refs,1):tex.append(r'\bibitem{ref'+str(i)+'} '+esc(ref)+r' \url{'+url+'}')
tex.extend([r'\end{thebibliography}',r'\end{document}'])
(ROOT/'paper.tex').write_text('\n'.join(tex)+'\n')
print(json.dumps({'pdf':str(PDF),'tex':str(ROOT/'paper.tex'),'blocks':len(m.blocks)}))
