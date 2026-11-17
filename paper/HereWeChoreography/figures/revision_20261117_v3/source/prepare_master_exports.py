"""Prepare one native scene pipeline for the paper and editable PowerPoint."""
from pathlib import Path
from lxml import etree as E
import shutil

ROOT=Path(__file__).resolve().parents[1]
OLD=ROOT.parent/'editable'
palette={'#203247':'#202C3C','#20384b':'#202C3C','#087e84':'#008F95','#167e83':'#008F95',
 '#277bb5':'#2474B5','#79549a':'#7852A9','#745493':'#7852A9','#bb831c':'#E69A28','#b08130':'#E69A28',
 '#eaf8f5':'#FAFCFB','#eaf3fb':'#FAFCFD','#f2edf9':'#FBFAFE','#fff5df':'#FFFCF7',
 '#edf5f6':'#FAFCFB','#f8fafc':'#FAFCFD','#f1ecf7':'#FBFAFE','#fffaf1':'#FFFCF7',
 '#e7f4f1':'#FAFCFB','#fbf9fd':'#FBFAFE','#94a5b1':'#B7BFC7','#b5c0c8':'#B7BFC7',
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
