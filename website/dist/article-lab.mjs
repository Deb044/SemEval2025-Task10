import {validateMatches} from './narrative-core.mjs';
const byId=id=>document.getElementById(id);
const el=(tag,text,cls)=> {const node=document.createElement(tag); if(text!==undefined)node.textContent=text; if(cls)node.className=cls; return node;};
const taxonomy=fetch('official-taxonomy.json',{cache:'no-store'}).then(response=> {if(!response.ok)throw new Error('Official taxonomy unavailable');return response.json();});
let worker, pending=false, controller, generatorReady=false;
const examples={
 climate:{domain:'CC',text:'The columnist argues that new climate levies will increase household bills and close small businesses. She says the policy will damage the economy rather than help it.'},
 war:{domain:'URW',text:'The commentator blames Western governments for provoking the invasion. He argues that NATO and its allies, rather than Russia, are responsible for starting the conflict.'},
 other:{domain:'CC',text:'The home team won the football match by two goals. Fans celebrated the result and the coach praised the players for their teamwork.'}
};
byId('overview-try').addEventListener('click',()=>byId('tab-analyze').click());
document.querySelectorAll('[data-official-example]').forEach(button=>button.addEventListener('click',()=> {
 const example=examples[button.dataset.officialExample]; byId('official-text').value=example.text; byId('official-domain').value=example.domain; byId('official-language').value='en';
 byId('official-status').textContent='Invented demonstration text loaded. Click Analyze article to run the model.';
}));
byId('official-file').addEventListener('change',async event=> {
 const file=event.target.files[0]; if(!file)return;
 try {const text=await file.text();if(text.length>20000)throw new Error('The file exceeds 20,000 characters.');byId('official-text').value=text;byId('official-status').textContent=`Loaded ${file.name} locally. Click Analyze article to run.`;}catch(error){byId('official-status').textContent=error.message;}
});
function updateMode(){byId('official-generate').disabled=byId('official-mode').value!=='local'||!generatorReady;if(byId('official-generate').disabled)byId('official-generate').checked=false;}
byId('official-mode').addEventListener('change',updateMode);
fetch('/api/status',{cache:'no-store'}).then(response=>response.json()).then(data=> {
 if(!data.local||!data.official_taxonomy)return;
 byId('official-local').disabled=false; generatorReady=data.generator;
 updateMode();
}).catch(()=>{});
function state(active){pending=active;byId('official-run').disabled=active;byId('official-cancel').hidden=!active;byId('official-mode').disabled=active;}
function progress(stage,detail){byId('official-status').textContent=`${stage}: ${detail}`;}
function trace(rows){
 const root=el('ol',undefined,'run-trace');
 rows.forEach(row=> {const item=el('li',undefined,'trace-'+row.state);item.append(el('strong',row.stage),el('span',row.state.replaceAll('_',' '),'badge'),el('p',row.detail));root.append(item);});return root;
}
function sourceView(text,evidence){
 const box=el('div',undefined,'source-text'); let cursor=0;
 const intervals=evidence.filter(unit=>text.slice(unit.start,unit.end)===unit.text).map(unit=>[unit.start,unit.end]).sort((a,b)=>a[0]-b[0]);
 const merged=[];intervals.forEach(pair=> {const previous=merged[merged.length-1];if(previous&&pair[0]<=previous[1])previous[1]=Math.max(previous[1],pair[1]);else merged.push([...pair]);});
 merged.forEach(([start,end])=> {box.append(document.createTextNode(text.slice(cursor,start)),el('mark',text.slice(start,end)));cursor=end;});box.append(document.createTextNode(text.slice(cursor)));return box;
}
async function result(data,text){
 validateMatches(await taxonomy,data.matches);
 const root=byId('official-output');root.replaceChildren();
 const panel=el('article',undefined,'panel');panel.append(el('p',data.engine,'eyebrow'),el('h3',data.matches.length===1?'Candidate official narrative':'Candidate official narratives'),el('p',data.note,'small'));
 const list=el('div',undefined,'narrative-matches');data.matches.forEach(match=> {
  const block=el('div');block.append(el('strong',match.label),el('p',`Parent: ${match.parent}`,'small'));
  block.append(el('p',Number.isFinite(match.score)?`Semantic similarity: ${match.score.toFixed(3)} · not a probability`:'No candidate passed the prototype selection checks.','small'));if(Number.isFinite(match.support_score))block.append(el('p',`Claim-agreement check: ${match.support_score.toFixed(3)} · model-based score`,'small'));list.append(block);
 });panel.append(list);
 panel.append(el('h4','Supporting source evidence'));const quotes=el('ol',undefined,'evidence');
 if(data.evidence.length)data.evidence.forEach(unit=>quotes.append(el('li',`[${unit.id}] ${unit.text}`)));else quotes.append(el('li','No supporting unit selected for Other.'));
 panel.append(quotes);
 if(data.generated_explanation){panel.append(el('h4',data.generated_explanation.fallback?'Extractive fallback':'Local Qwen draft · inspect against the source'),el('p',data.generated_explanation.text),el('p',`Target: ${data.generated_explanation.target}`,'small'));}
 else if(data.preview){panel.append(el('h4','Evidence extract · up to 80 words'),el('p',data.preview),el('p','This is source wording, not an LLM-written explanation.','small'));}
 if(data.explanation_error)panel.append(el('p',data.explanation_error,'small'));
 const source=el('details');source.append(el('summary','Inspect the article with selected evidence'),sourceView(text,data.evidence));panel.append(source);
 const ranking=el('details');ranking.append(el('summary','Inspect candidate ranking'));const ranked=el('ol');data.ranking.forEach(row=>ranked.append(el('li',`${row.label}: similarity ${row.score.toFixed(3)}${Number.isFinite(row.support_score)?`, support ${row.support_score.toFixed(3)}`:''}`)));ranking.append(ranked);panel.append(ranking);
 const protocol=el('p',`Taxonomy profile: ${data.taxonomy_version}`,'small');panel.append(protocol);root.append(panel);
 const flow=el('article',undefined,'panel');flow.append(el('h3','Pipeline trace'),trace(data.trace));root.append(flow);
 byId('official-status').textContent=`Analysis complete${Number.isFinite(data.seconds)?` in ${data.seconds.toFixed(1)} seconds`:''}. Inspect candidate labels and source evidence.`;
}
byId('official-form').addEventListener('submit',async event=> {
 event.preventDefault();if(pending)return;
 const text=byId('official-text').value.trim(),domain=byId('official-domain').value;
 if(!text||text.length>20000){progress('Input','Use 1–20,000 characters.');return;}
 state(true);byId('official-output').replaceChildren();
 try {
  await taxonomy;
  if(byId('official-mode').value==='local'){
   controller=new AbortController();progress('Local pipeline','Running the encoder and official-label matching. A detailed trace will appear when the calculation completes.');
   const response=await fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},signal:controller.signal,body:JSON.stringify({model:'semeval-official',text,domain,language:byId('official-language').value,max_labels:byId('official-more').checked?3:1,generate:byId('official-generate').checked})});
   const data=await response.json();if(!response.ok)throw new Error(data.error||'Local analysis failed');await result(data,text);state(false);
  }else{
   worker ||= new Worker(new URL('./narrative-worker.mjs',import.meta.url),{type:'module'});
   worker.onmessage=async ({data})=> {
    if(!pending)return;
    if(data.type==='progress'){progress(data.stage,data.detail);return;}
    try {if(data.type==='error')throw new Error(data.message);await result(data.result,text);}catch(error){progress('Analysis stopped',error.message);}finally{state(false);}
   };
   worker.onerror=error=> {progress('Browser model unavailable',error.message||'Check the model download connection or use the local workspace.');worker.terminate();worker=null;state(false);};
   progress('Starting','Preparing the browser encoder. The first run includes model downloads.');worker.postMessage({text,domain,maxLabels:byId('official-more').checked?3:1});
  }
 }catch(error){if(error.name!=='AbortError')progress('Analysis stopped',error.message);state(false);}
});
byId('official-cancel').addEventListener('click',()=> {if(worker){worker.terminate();worker=null;}controller?.abort();state(false);progress('Canceled','The current display request was canceled.');});
taxonomy.then(data=> {
 const root=byId('official-taxonomy-tree');root.replaceChildren();
 for(const parent of data.parents){const group=el('details');group.append(el('summary',parent.id));const children=el('ul');data.labels.filter(label=>label.parent===parent.id).forEach(label=>children.append(el('li',label.id)));group.append(children);root.append(group);}
}).catch(error=> {byId('official-taxonomy-tree').textContent=error.message;});
