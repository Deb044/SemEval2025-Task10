/* Shared, testable official-taxonomy matching. No article text is persisted. */
export const CUTOFF = 0.40;
export function splitUnits(text) {
  const units = [];
  const sentences = text.matchAll(/[^.!?।\r\n]+(?:[.!?।]+|$)/gu);
  for (const sentence of sentences) {
    const words = [...sentence[0].matchAll(/\S+/gu)];
    for (let index = 0; index < words.length; index += 32) {
      const block = words.slice(index, index + 32);
      const start = sentence.index + block[0].index;
      const last = block[block.length - 1];
      const end = sentence.index + last.index + last[0].length;
      units.push({id: units.length, text: text.slice(start, end), start, end});
    }
  }
  if (!units.length) throw new Error('Enter article text containing words.');
  if (units.length > 200) throw new Error('This article is too long for the interactive prototype.');
  return units;
}
export function labelQuery(label) {
  const prefix = label.domain === 'CC' ? 'Climate change. ' : 'Ukraine-Russia war. ';
  const parent = label.parent.split(': ').slice(1).join(': ');
  return prefix + parent + (label.is_other ? '' : '. ' + label.definition);
}
const dot = (a, b) => a.reduce((sum, value, index) => sum + value * b[index], 0);
export function matchLabels(taxonomy, labelVectors, textVectors, domain = 'CC', cutoff = CUTOFF) {
  const pooled = textVectors[0].map((_, index) => textVectors.reduce((sum, vector) => sum + vector[index], 0) / textVectors.length);
  const norm = Math.sqrt(dot(pooled, pooled)) || 1;
  pooled.forEach((value, index) => pooled[index] = value / norm);
  const eligible = taxonomy.labels.map((label, index) => ({label, index})).filter(({label}) => label.id !== 'Other' && (domain === 'all' || label.domain === domain));
  const scored = eligible.map(({label, index}) => {
    const similarities = textVectors.map(vector => dot(vector, labelVectors[index]));
    const best = Math.max(...similarities);
    const score = (.65 * best + .35 * dot(pooled, labelVectors[index])) * (label.is_other ? .9 : 1);
    return {label: label.id, parent: label.parent, score, evidence_id: similarities.indexOf(best), is_other: label.is_other};
  });
  const concreteParents = new Set(scored.filter(row => !row.is_other && row.score >= cutoff).map(row => row.parent));
  const ranked = scored.filter(row => !row.is_other || !concreteParents.has(row.parent)).sort((a,b) => b.score - a.score || a.label.localeCompare(b.label));
  const matches = ranked.filter(row => row.score >= cutoff).slice(0, 3);
  return {matches: matches.length ? matches : [{label:'Other',parent:'Other',score:null,evidence_id:null}], ranking: [...scored].sort((a,b)=>b.score-a.score||a.label.localeCompare(b.label)).slice(0,8)};
}
export function evidenceFor(units, matches) {
  const ids = new Set();
  matches.forEach(row => { if (row.evidence_id !== null) [row.evidence_id, row.evidence_id-1, row.evidence_id+1].forEach(id=> { if (id>=0 && id<units.length) ids.add(id); }); });
  const evidence = [...ids].sort((a,b)=>a-b).map(id=>units[id]);
  let words = 0; const preview = [];
  evidence.forEach(unit => { const count=unit.text.trim().split(/\s+/u).length; if(words+count<=80){ preview.push(unit.text); words+=count; } });
  return {evidence, preview: preview.join(' ')};
}
export function validateMatches(taxonomy, matches) {
  const lookup = new Map(taxonomy.labels.map(label=>[label.id,label.parent]));
  if (matches.some(row=>lookup.get(row.label)!==row.parent)) throw new Error('A result is outside the official taxonomy.');
}

export function supportedMatches(taxonomy, ranking, supports) {
  const lookup=new Map(taxonomy.labels.map(row=>[row.id,row]));
  const checked=ranking.map((row,index)=>{const value=supports[index];return {...row,entailment_score:value?.entailment??null,support_score:value?value.entailment/(value.entailment+value.contradiction||1):null};});
  checked.forEach(row=>row.selection_score=.8*row.score+.2*(row.support_score??0));
  const eligible=checked.filter(row=>Number.isFinite(row.support_score)&&row.support_score>=.6&&row.entailment_score>=.1&&row.score>=.3);
  const concrete=new Set(eligible.filter(row=>!lookup.get(row.label).is_other).map(row=>row.parent));
  const ordered=eligible.filter(row=>!lookup.get(row.label).is_other||!concrete.has(row.parent)).sort((a,b)=>b.selection_score-a.selection_score);
  const matches=ordered.filter(row=>row.selection_score>=ordered[0].selection_score-.05).slice(0,3);
  return {matches:matches.length?matches:[{label:'Other',parent:'Other',score:null,support_score:null,evidence_id:null}],ranking:checked.sort((a,b)=>b.selection_score-a.selection_score)};
}
