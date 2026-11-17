"""Prepare one native scene pipeline for the paper and editable PowerPoint."""
from pathlib import Path
from lxml import etree as E
import shutil

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'editable'
palette={'#203247':'#242A33','#20384b':'#242A33','#087e84':'#667A70','#167e83':'#667A70',
 '#277bb5':'#65778A','#79549a':'#85778C','#745493':'#85778C','#bb831c':'#A18C69','#b08130':'#A18C69',
 '#eaf8f5':'#FBFCFA','#eaf3fb':'#FAFBFD','#f2edf9':'#FCFAFD','#fff5df':'#FDFBFA',
 '#edf5f6':'#FBFCFA','#f8fafc':'#FAFBFD','#f1ecf7':'#FCFAFD','#fffaf1':'#FDFBFA',
 '#e7f4f1':'#FBFCFA','#fbf9fd':'#FCFAFD','#94a5b1':'#B7BFC7','#b5c0c8':'#B7BFC7',
 '#c5b085':'#B7BFC7','#9badb9':'#B7BFC7','#e66055':'#AD8B80'}
for name in ['agent_policy_comparison','dataset_construction']:
 tree=E.parse(str(OLD/(name+'.editable.svg')))
 for node in tree.iter():
  if not isinstance(node.tag,str):continue
  if E.QName(node).localname=='text':node.set('font-family','Times New Roman')
  for k in ['fill','stroke']:
   c=node.get(k,'').lower()
   if c in palette:node.set(k,palette[c])
 tree.write(str(ROOT/(name+'.svg')),encoding='utf-8',xml_declaration=False)

# The checked-in converter, builder and finalizer are canonical.
# Do not replace them with old bootstrap templates (which lack PDF fonts).
print('Prepared appendix SVGs; existing native export helpers retained.')
