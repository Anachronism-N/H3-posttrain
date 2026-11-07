"""Build a public-only, blank-rating review page without models or uploads."""
import argparse
import copy
import json
import os
from pathlib import Path
import re
import tempfile

from .editing import validate_edit_task
from .io import file_sha, object_sha, read_json_snapshot
from .reviews import blank_review_template, validate_review_pack
from .review_issues import ISSUE_TYPES, blank_issue_template
from .schema import keys, validate_plan, validate_song


_HTML = r'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>音乐编舞计划盲评</title>
<style>
body{font:15px sans-serif;max-width:1450px;margin:24px auto;padding:0 16px;line-height:1.6;background:#f6f7fb}
nav{position:sticky;top:0;background:#fff;padding:12px;z-index:3}a{margin-right:24px}audio{width:100%;margin:12px 0}
.columns{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:18px}.card{background:#fff;border:1px solid #ccd0db;border-radius:8px;padding:16px}
.timeline{height:60vh;overflow:auto}table{border-collapse:collapse;width:100%}td,th{border:1px solid #ddd;padding:8px;vertical-align:top}
tr.active{background:#fff0b0}input,select,button,textarea{font:inherit;margin:4px}textarea{width:95%;height:70px}.warn{background:#fff3d6;padding:12px}
pre{white-space:pre-wrap}@media(max-width:900px){.columns{grid-template-columns:1fr}}
</style>
<h1>音乐编舞计划盲评</h1>
<p class="warn">真实音频与文字计划，不是视频或实测人体动作。公共结构为工具预测，unknown不是人工真值。格式、覆盖、主题标签和文字差异不等于音乐贴合或可执行性。评分为空；无法判断请留空。</p>
<details id="rubric" open><summary>评分锚点与操作说明（校准草案，非已验证指标）</summary>
<p>先完整听取该case音频，再分别评价该case的匿名方案。各维度独立评分；无法判断留空并说明原因，不因entry/exit未知直接给最低分。无编辑任务的两项编辑分保持空值。</p>
<table><tr><th>维度</th><th>1分</th><th>3分</th><th>5分</th></tr>
<tr><td>音乐贴合</td><td>多处与听到的节奏/能量变化明显不合</td><td>局部贴合，整曲较通用</td><td>主要变化处有合理动作发展，安排清楚</td></tr>
<tr><td>整体组织/主题发展</td><td>缺少发展，重复难以解释</td><td>有主题与对比，发展较弱</td><td>主题回归、变奏、对比及结尾清楚</td></tr>
<tr><td>文字动作可执行性</td><td>多处含糊、冲突或明显超出所给时间</td><td>大体可理解，部分需澄清</td><td>动作与时间关系明确，可据此试排</td></tr>
<tr><td>衔接与边界</td><td>相邻动作多处断裂或冲突</td><td>大部分可衔接，部分依赖未说明条件</td><td>可判断的边界衔接明确、合理</td></tr></table>
<p>两名独立评审应各自完成本包所有已分配方案；每个维度均要核对两位的提交，不以每方案已有一份分数当全部完成。评审标识请用组织者预登记的匿名ID。本页未预登记真实评审者、不会提交到服务器。</p>
<p>评论记录具体原音频时间、公开事件ID与原文依据；区分合理主题回归和非预期复读。2分或以上分歧由组织者另行复核，不自动改分或判优。</p>
<pre id="comment-guide">time=01:12.0–01:20.0; events=event_0021,event_0022;
type=unintended_repeat; severity=major;
evidence=引用原动作并说明听到的音乐变化;
reason=为什么不是合理回归（或明确记录合理回归作为反例）。</pre>
<p>评论继续保留原格式；下方可另行手动记录并导出结构化时间观察。点击动作行只填时间/事件ID，不自动判断质量。合理主题回归与无法判断分别保留；空记录不代表无问题。听音勾选只代表本人声明，不证明真实播放完成。</p>
</details>
<nav id="cases"></nav><label>评审标识<input id="reviewer" placeholder="填写预登记匿名ID"></label>
<button id="export">导出评审JSON</button><button id="export-issues">另存时间问题JSON</button><main id="content"></main>
<script id="data" type="application/json">DATA</script>
<script>
const {pack,template,planningMusic,issueTemplate,issueTypes}=JSON.parse(document.getElementById('data').textContent);
const limitations=document.createElement('p');limitations.className='warn';
limitations.textContent=pack.limitations.join(' ');document.querySelector('nav').before(limitations);
const groups=new Map(),ids=[...new Set(pack.items.map(i=>i.song.song_id))].sort();
const E=(tag,text)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;return e};
const labels={musical_fit:'音乐贴合（实际听音）',long_range_structure:'整体组织/主题发展',planned_executability:'文字动作可执行性',planned_boundary_quality:'衔接与边界',edit_target_satisfaction:'编辑指令达成（无编辑留空）',edit_preservation:'编辑保持（无编辑留空）'};
ids.forEach((id,index)=>{
  const items=pack.items.filter(i=>i.song.song_id===id),group=E('div');
  group.id='case-'+(index+1);groups.set(id,group);
  const link=E('a','Case '+(index+1));link.href='#'+group.id;
  document.getElementById('cases').append(link);
  group.append(E('h2','Case '+(index+1)+' · '+items.length+'个匿名方案'));
  const audio=E('audio');audio.controls=true;audio.preload='metadata';audio.src=items[0].song.audio.path;
  group.append(audio);
  const speed=E('select');[.5,.75,1,1.25,1.5].forEach(v=>speed.append(new Option(v+'倍速',v)));
  speed.value='1';speed.onchange=()=>audio.playbackRate=Number(speed.value);group.append(speed);
  const locate=E('button','定位当前音频对应动作');group.append(locate,E('p',items[0].song.brief));
  const detail=E('details');detail.append(E('summary','原dataset公开结构'),E('pre',JSON.stringify(items[0].song.music,null,2)));group.append(detail);
  const planning=E('details');planning.hidden=!planningMusic[id];planning.append(E('summary','两方案生成器共享的预测结构（非人工真值）'),E('pre',JSON.stringify(planningMusic[id],null,2)));group.append(planning);
  const columns=E('div');columns.className='columns';group.append(columns);
  items.forEach(item=>{
    const review=template.reviews.find(r=>r.review_id===item.review_id),card=E('section');card.className='card';
    card.append(E('h3',item.review_id));
    if(item.edit_task){const edit=E('details');edit.append(E('summary','编辑指令与原计划'),E('p',item.edit_task.instruction),E('pre',JSON.stringify(item.source_plan,null,2)));card.append(edit)}
    const checkbox=E('input');checkbox.type='checkbox';checkbox.onchange=()=>review.audio_seen=checkbox.checked;
    const heard=E('label','我已实际听取该方案对应音频');heard.prepend(checkbox);card.append(heard);
    const comment=E('textarea');comment.placeholder='原音频时间、公开事件ID、原文依据、问题或无法判断原因';comment.oninput=()=>review.comments=comment.value;
    const issueRecord=issueTemplate.records.find(r=>r.review_id===item.review_id);
    const observations=E('details');observations.className='observations';observations.append(E('summary','结构化时间观察（单独导出，不计质量分）'));
    const issueType=E('select');issueType.className='issue-type';issueType.append(new Option('选择观察类型',''));
    Object.entries(issueTypes).forEach(([key,label])=>issueType.append(new Option(label,key)));
    const severity=E('select');severity.className='issue-severity';severity.append(new Option('选择程度',''));
    [['minor','轻微'],['major','严重'],['not_applicable','非问题/合理回归'],['uncertain','无法判断']].forEach(([key,label])=>severity.append(new Option(label,key)));
    const start=E('input'),end=E('input'),events=E('input'),evidence=E('textarea'),reason=E('textarea');
    start.className='issue-start';end.className='issue-end';events.className='issue-events';
    start.placeholder='起点（整数采样点）';end.placeholder='终点（整数采样点）';events.placeholder='公开事件ID，逗号分隔';
    evidence.className='issue-evidence';reason.className='issue-reason';evidence.placeholder='引用实际动作与音乐观察依据';reason.placeholder='判断理由；复读须区分合理主题回归';
    observations.append(E('p','时间使用本计划的原音频采样时钟，半开[start,end)。不填默认类型或程度；每个引用事件须与所填时间相交。'),issueType,severity,start,end,events,evidence,reason);
    const addIssue=E('button','保存本条观察'),issueList=E('div');addIssue.className='add-issue';issueList.className='issue-list';let issueCounter=0;
    function redrawIssues(){
      issueList.replaceChildren();issueRecord.issues.forEach(issue=>{
        const entry=E('div'),remove=E('button','移除此条草稿');entry.append(E('pre',JSON.stringify(issue,null,2)));
        remove.onclick=()=>{issueRecord.issues=issueRecord.issues.filter(x=>x!==issue);redrawIssues()};entry.append(remove);issueList.append(entry);
      });
    }
    addIssue.onclick=()=>{
      const a=Number(start.value),b=Number(end.value),ids=events.value.split(',').map(v=>v.trim());
      const byId=new Map(item.plan.events.map(e=>[e.event_id,e]));
      if(!/^(0|[1-9][0-9]*)$/.test(start.value)||!/^(0|[1-9][0-9]*)$/.test(end.value)||!Number.isSafeInteger(a)||!Number.isSafeInteger(b)||a>=b||b>item.plan.num_samples){alert('需要音频范围内的非空整数采样点区间');return}
      if(!ids.length||new Set(ids).size!==ids.length||ids.some(id=>!byId.has(id)||byId.get(id).start>=b||a>=byId.get(id).end)){alert('每个公开事件ID必须已存在、唯一且与观察时间相交');return}
      if(!issueType.value||!severity.value||!evidence.value.trim()||!reason.value.trim()){alert('请独立填写类型、程度、依据和理由');return}
      const permitted=issueType.value==='reasonable_recurrence'?['not_applicable']:issueType.value==='uncertain'?['uncertain']:['minor','major'];
      if(!permitted.includes(severity.value)){alert('合理主题回归使用非问题程度，无法判断使用无法判断程度');return}
      if(issueType.value==='music_mismatch'&&!checkbox.checked){alert('音乐错配观察需要本人听音声明');return}
      issueRecord.issues.push({issue_id:'issue_'+(++issueCounter),type:issueType.value,severity:severity.value,start:a,end:b,event_ids:ids,evidence:evidence.value,reason:reason.value});
      evidence.value='';reason.value='';issueType.value='';severity.value='';redrawIssues();
    };
    observations.append(addIssue,issueList);
    const timeline=E('div');timeline.className='timeline';const table=E('table'),header=E('tr');
    ['事件ID','时间（秒）','轨道','动作'].forEach(t=>header.append(E('th',t)));table.append(header);
    item.plan.events.forEach(e=>{
      const row=E('tr');row.dataset.start=e.start/item.plan.sample_rate;row.dataset.end=e.end/item.plan.sample_rate;
      row.append(E('td',e.event_id),E('td',Number(row.dataset.start).toFixed(4)+'–'+Number(row.dataset.end).toFixed(4)),E('td',e.track),E('td',e.action));
      row.title='添加本事件的时间与ID到评论（不自动判断质量）';
      row.onclick=()=>{start.value=e.start;end.value=e.end;events.value=e.event_id;comment.value+=(comment.value?' | ':'')+'time='+row.dataset.start+'–'+row.dataset.end+'s; samples=['+e.start+','+e.end+'); events='+e.event_id+'; evidence=';review.comments=comment.value};
      table.append(row);
    });
    timeline.append(table);card.append(timeline);
    const motifs=E('details');motifs.append(E('summary','声明的主题（非质量证据）'),E('pre',JSON.stringify(item.plan.motifs,null,2)));card.append(motifs);
    for(const [key,label] of Object.entries(labels)){
      const field=E('label',label),select=E('select');select.append(new Option('未评审/不适用',''));
      for(let n=1;n<=5;n++)select.append(new Option(n,n));
      select.disabled=key.startsWith('edit_')&&!item.edit_task;
      select.onchange=()=>review.ratings[key]=select.value?Number(select.value):null;field.append(select);card.append(field);
    }
    card.append(comment,observations);columns.append(card);
  });
  function highlight(){
    columns.querySelectorAll('tr[data-start]').forEach(row=>row.classList.toggle('active',Number(row.dataset.start)<=audio.currentTime&&audio.currentTime<Number(row.dataset.end)));
  }
  audio.addEventListener('timeupdate',highlight);audio.addEventListener('seeked',highlight);
  locate.onclick=()=>{
    highlight();columns.querySelectorAll('.timeline').forEach(t=>t.querySelector('tr.active')?.scrollIntoView({block:'nearest',behavior:'smooth'}));
  };
  document.getElementById('content').append(group);
});
function show(){
  const requested=Number((location.hash.match(/^#case-([0-9]+)$/)||[])[1]||1)-1;
  const index=Number.isInteger(requested)&&requested>=0&&requested<ids.length?requested:0;
  groups.forEach((group,id)=>{group.hidden=id!==ids[index];if(group.hidden)group.querySelector('audio').pause()});
}
addEventListener('hashchange',show);show();
document.getElementById('export').onclick=()=>{
  const id=document.getElementById('reviewer').value.trim();if(!id||id==='FILL_REVIEWER_ID'){alert('请填写实际预登记的匿名评审标识');return}
  if(template.reviews.some(r=>r.ratings.musical_fit!==null&&!r.audio_seen)){alert('音乐贴合评分需要本人听音声明；未听音请保留空值');return}
  template.reviews.forEach(r=>r.reviewer_id=id);
  const blob=new Blob([JSON.stringify(template,null,2)],{type:'application/json'}),link=E('a');
  link.href=URL.createObjectURL(blob);link.download='reviews.completed.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
};
document.getElementById('export-issues').onclick=()=>{
  const id=document.getElementById('reviewer').value.trim();if(!id||id==='FILL_REVIEWER_ID'){alert('请填写实际预登记的匿名评审标识');return}
  if(issueTemplate.records.some(r=>r.issues.some(i=>i.type==='music_mismatch')&&!template.reviews.find(v=>v.review_id===r.review_id).audio_seen)){alert('音乐错配观察需要本人听音声明');return}
  issueTemplate.records.forEach(r=>{r.reviewer_id=id;r.audio_seen=template.reviews.find(v=>v.review_id===r.review_id).audio_seen});
  const blob=new Blob([JSON.stringify(issueTemplate,null,2)],{type:'application/json'}),link=E('a');
  link.href=URL.createObjectURL(blob);link.download='review-issues.json';link.click();setTimeout(()=>URL.revokeObjectURL(link.href),1000);
};
</script></html>'''


def _public_songs(pack):
    songs = {}
    for item in pack['items']:
        song = item['song']
        keys(song, ('song_id', 'brief', 'audio', 'music'), where='public song')
        probe = dict(copy.deepcopy(song), split='dev', song_group_id=None,
                     artist_group_id=None, near_duplicate_group_id=None)
        validate_song(probe)
        validate_plan(item['plan'], probe)
        source, task = item['source_plan'], item['edit_task']
        if (source is None) != (task is None):
            raise ValueError('Edit review requires both public source plan and task')
        if task is not None:
            validate_plan(source, probe)
            if source['provenance'] or any(e['evidence'] for e in source['events']):
                raise ValueError('Public source plan contains private provenance or citations')
            keys(task, ('schema', 'task_id', 'song_id', 'source_plan_sha256', 'instruction',
                        'allowed_event_ids', 'repair_event_ids', 'protected'), where='public edit task')
            if task['schema'] != 'choreobench.public_edit_task.v1':
                raise ValueError('Only public_edit_task.v1 may enter the review page')
            validate_edit_task(dict(copy.deepcopy(task), schema='choreobench.edit_task.v1',
                                    goals=[], semantic_goals=[]), source)
        route = song['audio']['path']
        if not re.fullmatch(r'assets/[A-Za-z0-9_.-]+', route) or route.split('/')[1] in ('.', '..'):
            raise ValueError('Review audio must use a relative assets/FILENAME route')
        sid = song['song_id']
        if sid in songs and object_sha(songs[sid]) != object_sha(song):
            raise ValueError('Same public song ID has inconsistent audio/clock/music')
        songs[sid] = song
    return songs


def render_review_page(pack, planning_music=None):
    """Render only the validated public package, always with fresh null ratings.

    Optional planning music is a public map from every anonymous song ID to
    a strict music object on the same clock. It is shown separately, never
    inserted into the hash-bound scoring package or used to compute scores.
    """
    validate_review_pack(pack)
    songs = _public_songs(pack)
    music = {} if planning_music is None else copy.deepcopy(planning_music)
    if planning_music is not None:
        if not isinstance(music, dict) or set(music) != set(songs):
            raise ValueError('Planning music must cover exactly the anonymous public songs')
        for sid, value in music.items():
            probe = dict(copy.deepcopy(songs[sid]), music=value, split='dev', song_group_id=None,
                         artist_group_id=None, near_duplicate_group_id=None)
            validate_song(probe)
    payload = dict(pack=pack, template=blank_review_template(pack), planningMusic=music,
                   issueTemplate=blank_issue_template(pack), issueTypes=ISSUE_TYPES)
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace('<', chr(92) + 'u003c')
    return _HTML.replace('DATA', encoded, 1)


def generate_review_page(public_path, out_path, planning_music_path=None):
    """Exclusively publish HTML; never read bindings, grades, audio or credentials."""
    public_path, out_path = Path(public_path), Path(out_path)
    if out_path.exists() or out_path.is_symlink():
        raise FileExistsError(str(out_path))
    pack, digest = read_json_snapshot(public_path)
    sources = [(public_path, digest)]
    music = None
    if planning_music_path is not None:
        music, music_sha = read_json_snapshot(planning_music_path)
        sources.append((Path(planning_music_path), music_sha))
    page = render_review_page(pack, music)
    for path, expected in sources:
        if file_sha(path) != expected:
            raise ValueError('Review page input changed while rendering')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', newline=chr(10),
                dir=out_path.parent, suffix='.tmp', delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(page)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(str(temporary), str(out_path))
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return dict(schema='choreobench.review_page.v1', public_pack_sha256=object_sha(pack),
        public_file_sha256=digest, page_file_sha256=file_sha(out_path),
        planning_music_file_sha256=sources[1][1] if music is not None else None,
        review_items=len(pack['items']), songs=len({i['song']['song_id'] for i in pack['items']}),
        initial_ratings='null', new_model_calls=0, quality_verified=False)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--public', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--planning-music', type=Path)
    args = parser.parse_args(argv)
    report = generate_review_page(args.public, args.out, args.planning_music)
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))
    return report


if __name__ == '__main__':
    main()
