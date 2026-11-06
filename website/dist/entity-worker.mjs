import {targets,context,selectRoles} from './entity-core.mjs';
let runtime;
const update=(stage,detail)=>self.postMessage({type:'progress',stage,detail});
async function prepare(){
 runtime ||= (async()=>{
  const [taxonomy,config]=await Promise.all([fetch('./entity-roles.json').then(r=>r.json()),fetch('./official-taxonomy.json').then(r=>r.json())]);
  update('Role model','Loading the pinned multilingual NLI transformer; article text stays on this device.');
  const {AutoTokenizer,AutoModelForSequenceClassification,env}=await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');env.allowLocalModels=false;env.backends.onnx.wasm.numThreads=1;
  const settings={revision:config.entailment_model.revision,quantized:true,progress_callback:event=>{if(event.status==='progress')update('Role model',`${event.file}: ${Math.round(event.progress||0)}%`);}};
  const [tokenizer,model]=await Promise.all([AutoTokenizer.from_pretrained(config.entailment_model.model,settings),AutoModelForSequenceClassification.from_pretrained(config.entailment_model.model,settings)]);
  if(model.config.label2id.entailment!==0)throw new Error('Unexpected entailment-label order.');
  return {taxonomy,tokenizer,model};
 })().catch(error=>{runtime=null;throw error;});return runtime;
}
self.onmessage=async event=>{try{
 const payload=event.data;const rows=targets(payload.text,payload.mentions,payload.article_id,payload.language);
 const {taxonomy,tokenizer,model}=await prepare();const predictions=[];
 for(let index=0;index<rows.length;index++){
  const row=rows[index],snippet=context(row),values=[];
  for(let i=0;i<taxonomy.labels.length;i++){
   const role=taxonomy.labels[i];update('Role checks',`Target ${index+1}/${rows.length} · role ${i+1}/22`);
   const premise=snippet.premise;
   const input=tokenizer(premise,{text_pair:role.hypothesis,truncation:true,max_length:384,padding:true});
   const output=await model(input),logits=Array.from(output.logits.data),maximum=Math.max(...logits);const p=logits.map(x=>Math.exp(x-maximum));const sum=p.reduce((a,b)=>a+b,0);p.forEach((x,j)=>p[j]=x/sum);values.push(p[0]);
  }
  const selected=selectRoles(taxonomy,values);predictions.push({...row,...selected,evidence:snippet});
 }
 self.postMessage({type:'result',data:{predictions,mode:'browser-nli',note:'Zero-shot role candidates from a multilingual NLI transformer; no GNN is run in this browser mode. Neutral or weakly supported targets can abstain. Scores are not calibrated probabilities.',
  trace:[{stage:'Supplied targets',detail:'Exact occurrences validated; no entity detector substituted other mentions.'},{stage:'Targeted source context',detail:'Only the supplied target occurrence is normalized inside its source sentence; attribution/negation require inspection.'},{stage:'Cross-lingual model',detail:'Pinned multilingual MiniLMv2 NLI scores all 22 role descriptions.'},{stage:'Hierarchy',detail:'One parent; compatible fine roles within 85% of the strongest score and above an exploratory cutoff. No invented Other role.'}]}});
 }catch(error){self.postMessage({type:'error',message:error.message});}};
