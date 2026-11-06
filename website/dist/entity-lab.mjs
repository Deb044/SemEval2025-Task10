import {targets,toTSV} from './entity-core.mjs';
const $=id=>document.getElementById(id);const node=(tag,text)=>{const n=document.createElement(tag);if(text!==undefined)n.textContent=text;return n;};
let selected=[],worker,last=[],pending=false,controls=[];
function occurrences(){
 const text=$('entity-text').value,mention=$('entity-name').value;const select=$('entity-occurrence');select.replaceChildren();
 if(!mention)return;
 for(let start=text.indexOf(mention);start>=0;start=text.indexOf(mention,start+1)){
  const option=node('option',`Occurrence ${select.options.length+1} · codepoint ${[...text.slice(0,start)].length}`);option.value=String(start);select.append(option);
 }
}
function list(){const root=$('entity-targets');root.replaceChildren();selected.forEach((target,i)=>{const li=node('li',`${target.mention} · browser offsets ${target.start}–${target.end} `);const remove=node('button','Remove');remove.type='button';remove.addEventListener('click',()=>{selected.splice(i,1);list();});li.append(remove);root.append(li);});}
$('entity-text').addEventListener('input',()=>{selected=[];list();occurrences();});$('entity-name').addEventListener('input',occurrences);
$('entity-add').addEventListener('click',()=>{try{
 const start=Number($('entity-occurrence').value),mention=$('entity-name').value;
 if(!$('entity-occurrence').options.length)throw new Error('Enter an exact entity string found in the article.');
 const values=[...selected,{mention,start,end:start+mention.length}];targets($('entity-text').value,values,$('entity-id').value,$('entity-language').value);selected=values;list();$('entity-status').textContent='Target added. Add another occurrence or run framing.';
 }catch(error){$('entity-status').textContent=error.message;}});
$('entity-example').addEventListener('click',()=>{
 $('entity-text').value='Mira rescued trapped families from the flood. Oleg stole the relief funds. The flood injured Nila.';$('entity-language').value='en';$('entity-name').value='Mira';selected=[];
 for(const mention of ['Mira','Oleg','Nila']){const start=$('entity-text').value.indexOf(mention);selected.push({mention,start,end:start+mention.length});}occurrences();list();$('entity-status').textContent='Invented example loaded. These examples are interface checks, not benchmark data.';
});
fetch('/api/status').then(r=>r.json()).then(data=>{
 if(data.local){$('entity-local').disabled=false;if(data.entity_gnn)$('entity-gcn').disabled=false;}
}).catch(()=>{});
function render(data){
 const root=$('entity-output');root.replaceChildren(node('p',data.note));last=data.predictions;
 if(data.training?.covered_fine_roles!==undefined)root.append(node('p',`Local training coverage: ${data.training.covered_fine_roles}/22 fine roles. Labels: ${data.training.annotation_sources.join('; ')}.`));
 for(const row of data.predictions){
  const block=node('article');block.className='panel';block.append(node('h4',`${row.mention} · codepoint offsets ${row.start}–${row.end} (exclusive)`));
  block.append(node('p',row.abstained?'Insufficient evidence for a supported role.':`${row.main_role}: ${row.labels.join(', ')}`));
  block.append(node('p',`Model score ${row.confidence.toFixed(3)} · not a truth probability`));
  const a=[...$('entity-text').value];const source=node('p');source.className='source-text';source.append(document.createTextNode(a.slice(row.evidence.start,row.start).join('')));const mark=node('mark',row.mention);source.append(mark,document.createTextNode(a.slice(row.end,row.evidence.end).join('')));block.append(source);
  if(row.ranking){const details=node('details');details.append(node('summary','Inspect all role scores'));const list=node('ol');row.ranking.forEach(r=>list.append(node('li',`${r.parent} / ${r.label}: ${r.score.toFixed(3)}`)));details.append(list);block.append(details);}
  root.append(block);
 }
 const trace=node('ol');(data.trace||[]).forEach(stage=>trace.append(node('li',`${stage.stage}: ${stage.detail}`)));root.append(trace);$('entity-download').disabled=!last.length||last.some(r=>r.abstained||!r.labels.length);
}
$('entity-form').addEventListener('submit',async event=>{
 event.preventDefault();if(pending)return;
 try{
  targets($('entity-text').value,selected,$('entity-id').value,$('entity-language').value);
  controls=[...$('entity-form').querySelectorAll('input,select,textarea,button')].map(element=>[element,element.disabled]);controls.forEach(([element])=>element.disabled=true);
  pending=true;$('entity-run').disabled=true;$('entity-download').disabled=true;$('entity-status').textContent='Running entity framing…';
  const mode=$('entity-mode').value,payload={text:$('entity-text').value,article_id:$('entity-id').value,language:$('entity-language').value,mentions:selected,offset_unit:'utf16',offset_end:'exclusive',mode};
  if(mode==='browser'){
   worker ||= new Worker('./entity-worker.mjs',{type:'module'});
   await new Promise((resolve,reject)=>{worker.onmessage=event=>{const m=event.data;if(m.type==='progress')$('entity-status').textContent=`${m.stage}: ${m.detail}`;else if(m.type==='result'){render(m.data);resolve();}else if(m.type==='error')reject(new Error(m.message));};worker.onerror=()=>reject(new Error('The role worker failed. Reload and try again.'));worker.postMessage(payload);});
  }else{
   const response=await fetch('/api/entity',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)});const data=await response.json();if(!response.ok)throw new Error(data.error||'Local framing failed');render(data);
  }
  $('entity-status').textContent='Entity framing complete. Inspect source portrayal and evidence before using these predictions.';
 }catch(error){$('entity-status').textContent=error.message;}
 finally{pending=false;controls.forEach(([element,disabled])=>element.disabled=disabled);$('entity-run').disabled=false;}
});
$('entity-download').addEventListener('click',()=>{try{
 const text=toTSV(last,$('entity-export-end').value);const url=URL.createObjectURL(new Blob([text],{type:'text/tab-separated-values;charset=utf-8'}));const a=node('a');a.href=url;a.download='subtask1-predictions.tsv';a.click();URL.revokeObjectURL(url);
 }catch(error){$('entity-status').textContent=error.message;}});
