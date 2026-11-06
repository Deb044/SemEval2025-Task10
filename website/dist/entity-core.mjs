export const parents=['Antagonist','Innocent','Protagonist'];
export function targets(text,values,articleId='user-input.txt',language='en'){
 if(typeof text!=='string'||!text.trim()||text.length>50000||!['en','bg','hi','pt','ru'].includes(language))throw new Error('Enter an article of up to 50,000 characters and choose a task language.');
 if(!values.length||values.length>30)throw new Error('Add 1–30 supplied target mentions.');
 if(!articleId||/[\t\r\n]/.test(articleId))throw new Error('Use an article ID without tabs or newlines.');
 const seen=new Set();
 return values.map(value=>{
  const {start,end,mention}=value;
  if(!Number.isInteger(start)||!Number.isInteger(end)||start<0||end<=start||end>text.length||text.slice(start,end)!==mention)throw new Error('A target no longer matches the exact article text. Add it again.');
  if((start>0&&/[\uD800-\uDBFF]/u.test(text[start-1])&&/[\uDC00-\uDFFF]/u.test(text[start]))||(end<text.length&&/[\uD800-\uDBFF]/u.test(text[end-1])&&/[\uDC00-\uDFFF]/u.test(text[end])))throw new Error('Target boundary splits a Unicode character.');
  const key=`${start}:${end}`;if(seen.has(key))throw new Error('The same occurrence was added twice.');seen.add(key);
  return {task:1,text,article_id:articleId,language,mention,start:[...text.slice(0,start)].length,end:[...text.slice(0,end)].length,start16:start,end16:end};
 });
}
export function context(row){
 const chars=[...row.text];const before=chars.slice(0,row.start).join(''),after=chars.slice(row.end).join('');
 const boundaries=[...before.matchAll(/[.!?।\n]/gu)];const last=boundaries.at(-1);
 const next=after.search(/[.!?।\n]/u);
 const a=Math.max(last?[...before.slice(0,last.index+1)].length:0,row.start-250);
 const b=Math.min(next>=0?row.end+[...after.slice(0,next+1)].length:chars.length,row.end+250);
 return {text:chars.slice(a,b).join(''),premise:chars.slice(a,row.start).join('')+'The target'+chars.slice(row.end,b).join(''),start:a,end:b};
}
export function selectRoles(taxonomy,values){
 const broad=parents.map(parent=>Math.max(...taxonomy.labels.map((role,i)=>role.parent===parent?values[i]:0)));
 const parent=parents[broad.indexOf(Math.max(...broad))];
 const ranking=taxonomy.labels.map((role,i)=>({label:role.id,parent:role.parent,score:values[i]})).sort((a,b)=>b.score-a.score);
 const choices=ranking.filter(r=>r.parent===parent);const threshold=Math.max(.25,choices[0].score*.85);
 const selected=choices.filter(r=>r.score>=threshold);
 return {main_role:selected.length?parent:null,labels:selected.map(r=>r.label),abstained:!selected.length,confidence:choices[0].score,ranking};
}
export function toTSV(rows,ending='exclusive'){
 if(!['exclusive','inclusive'].includes(ending))throw new Error('Select an end-offset convention.');
 return rows.map(row=>{
  if(!row.labels.length||row.abstained||!row.main_role)throw new Error('An abstention cannot be exported as an official prediction.');
  const fields=[row.article_id,row.mention,String(row.start),String(row.end-(ending==='inclusive'?1:0)),row.main_role,...row.labels];
  if(fields.some(field=>/[\t\r\n]/.test(field)))throw new Error('TSV fields cannot contain tabs or newlines.');
  return fields.join('\t');
 }).join('\n')+'\n';
}
