import {splitUnits, labelQuery, matchLabels, evidenceFor, validateMatches, supportedMatches} from './narrative-core.mjs';
let encoderPromise, taxonomyPromise, labelVectors, nliPromise;
const update = (stage, detail) => self.postMessage({type:'progress',stage,detail});
async function prepare() {
  taxonomyPromise ||= fetch('./official-taxonomy.json').then(response=> { if(!response.ok) throw new Error('Taxonomy unavailable'); return response.json(); });
  const taxonomy = await taxonomyPromise;
  if (!encoderPromise) encoderPromise = (async()=> {
    update('Loading encoder','First run downloads model/runtime files (about 280 MB); subsequent runs reuse the browser cache.');
    const {pipeline, env} = await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');
    env.allowLocalModels = false;
    env.backends.onnx.wasm.numThreads = 1;
    return pipeline('feature-extraction', taxonomy.browser_encoder.model, {
      revision:taxonomy.browser_encoder.revision, quantized:true,
      progress_callback:event=> { if(event.status==='progress' && event.progress!==undefined) update('Loading encoder', `${event.file}: ${Math.round(event.progress)}%`); }
    });
  })().catch(error=> {encoderPromise=null; throw error;});
  const encoder = await encoderPromise;
  if (!labelVectors) {
    const built=[];
    for(let index=0;index<taxonomy.labels.length;index+=4){
      update('Label embeddings',`${Math.min(index+4,taxonomy.labels.length)}/${taxonomy.labels.length} official label descriptions`);
      const tensor=await encoder(taxonomy.labels.slice(index,index+4).map(labelQuery),{pooling:'mean',normalize:true,truncation:true,max_length:128});
      built.push(...tensor.tolist());
    }
    labelVectors=built;
  }
  return {taxonomy,encoder};
}
async function checkSupport(taxonomy, units, ranking) {
  nliPromise ||= (async()=> {
    update('Entailment model','Loading a compact multilingual model to check candidate claims against source context.');
    const {AutoTokenizer,AutoModelForSequenceClassification,env}=await import('https://cdn.jsdelivr.net/npm/@xenova/transformers@2.17.2');
    env.allowLocalModels=false;env.backends.onnx.wasm.numThreads=1;
    const settings={revision:taxonomy.entailment_model.revision,quantized:true,
      progress_callback:event=>{if(event.status==='progress'&&event.progress!==undefined)update('Entailment model',`${event.file}: ${Math.round(event.progress)}%`);}};
    const [tokenizer,model]=await Promise.all([AutoTokenizer.from_pretrained(taxonomy.entailment_model.model,settings),AutoModelForSequenceClassification.from_pretrained(taxonomy.entailment_model.model,settings)]);
    if(model.config.label2id.entailment!==0)throw new Error('Unexpected NLI label order');
    return {tokenizer,model};
  })().catch(error=>{nliPromise=null;throw error;});
  const {tokenizer,model}=await nliPromise;
  const labels=new Map(taxonomy.labels.map(row=>[row.id,row]));const supports=[];
  for(let index=0;index<ranking.length;index++){
    const row=ranking[index];update('Evidence support check',`${index+1}/${ranking.length} candidate claims`);
    if(row.score<.3){supports.push(null);continue;}
    const premise=units.slice(Math.max(0,row.evidence_id-1),row.evidence_id+2).map(unit=>unit.text).join(' ');
    const inputs=tokenizer(premise,{text_pair:labels.get(row.label).hypothesis,padding:true,truncation:true,max_length:384});
    const output=await model(inputs);const values=Array.from(output.logits.data);const max=Math.max(...values);
    const probabilities=values.map(value=>Math.exp(value-max));const total=probabilities.reduce((a,b)=>a+b,0);supports.push({entailment:probabilities[0]/total,contradiction:probabilities[2]/total});
  }
  return supports;
}
self.onmessage=async event=> {
  try {
    const {text,domain}=event.data;
    if(typeof text!=='string'||!text.trim()||text.length>20000||!['CC','URW','all'].includes(domain)) throw new Error('Use 1–20,000 characters and select a taxonomy domain.');
    const started=performance.now(), units=splitUnits(text);
    update('Input',`${text.length} characters split into ${units.length} text units.`);
    const {taxonomy,encoder}=await prepare();
    const vectors=[];
    for(let index=0;index<units.length;index+=4){
      update('Article embeddings',`${Math.min(index+4,units.length)}/${units.length} text units`);
      const tensor=await encoder(units.slice(index,index+4).map(unit=>unit.text),{pooling:'mean',normalize:true,truncation:true,max_length:128});
      vectors.push(...tensor.tolist());
    }
    update('Candidate ranking','Matching article embeddings to official narrative/subnarrative label descriptions.');
    const retrieved=matchLabels(taxonomy,labelVectors,vectors,domain);
    const supports=await checkSupport(taxonomy,units,retrieved.ranking);
    const selected=supportedMatches(taxonomy,retrieved.ranking,supports);
    validateMatches(taxonomy,selected.matches);
    const extracts=evidenceFor(units,selected.matches);
    const trace=[
      {stage:'Input',state:'complete',detail:`${text.length} characters; ${units.length} text units of at most 32 words.`},
      {stage:'Multilingual encoder',state:'complete',detail:`Quantized multilingual MiniLM; ${vectors[0].length}-dimensional normalized vectors. Text is processed in this browser.`},
      {stage:'Candidate ranking',state:'complete',detail:'Cosine-based retrieval of up to eight official-label candidates.'},
      {stage:'Entailment check',state:'complete',detail:'Multilingual MiniLMv2 NLI checks candidate claims against nearby source context; Entailment-vs-contradiction agreement ≥0.70 and three-way entailment ≥0.25 are exploratory cutoffs, not official-task calibration.'},
      {stage:'Hierarchy validation',state:'complete',detail:'Returned names and parent links are restricted to the 96-label organizer taxonomy profile.'},
      {stage:'Label GNN',state:'not_used',detail:'Auxiliary GNN checkpoints are not an official-taxonomy classifier and are not used in this browser mode.'},
      {stage:'Evidence',state:'complete',detail:`${extracts.evidence.length} source units retained with neighboring context.`},
      {stage:'Explanation',state:'extractive',detail:'Source-evidence extract of at most 80 words. No LLM generation is performed in browser mode.'}
    ];
    self.postMessage({type:'result',result:{...selected,...extracts,units,trace,seconds:(performance.now()-started)/1000,
      engine:'Browser MiniLM retrieval + multilingual NLI · zero-shot',source:taxonomy.source,taxonomy_version:taxonomy.version,
      note:'Official-vocabulary candidates, not validated official-task predictions. Inspect their source evidence.',generated_explanation:null}});
  } catch(error){self.postMessage({type:'error',message:error.message});}
};
