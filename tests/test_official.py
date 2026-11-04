"""Official taxonomy and evidence invariants; fixtures are not benchmark results."""
import unittest
import numpy as np
from semeval.official import taxonomy, units, select_matches, extract_evidence


class OfficialTaxonomyChecks(unittest.TestCase):
    def test_inventory_is_complete_and_has_explicit_parent_links(self):
        data = taxonomy()
        self.assertEqual(len(data['labels']), 96)
        self.assertEqual(len(data['parents']), 22)
        parents={row['id'] for row in data['parents']}
        self.assertTrue(all(row['parent'] in parents for row in data['labels']))
        self.assertIn('CC: Criticism of climate policies: Climate policies have negative impact on the economy', {row['id'] for row in data['labels']})
        self.assertFalse(any(row['id'].startswith(('4_', 'Loaded_')) for row in data['labels']))
        self.assertEqual(sum(row['id'] == 'Other' for row in data['labels']), 1)

    def test_outputs_are_domain_restricted_and_parent_other_is_supported(self):
        data=taxonomy(); vectors=np.zeros((96,2),dtype=np.float32)
        target=next(i for i,row in enumerate(data['labels']) if row['id']=='URW: Hidden plots by secret schemes of powerful groups: Other')
        vectors[target]=[1,0]
        matches,_=select_matches(data,vectors,np.array([[1.,0.]]),'URW')
        self.assertEqual(matches[0]['label'],data['labels'][target]['id'])
        matches,_=select_matches(data,vectors,np.array([[1.,0.]]),'CC')
        self.assertEqual(matches[0]['label'],'Other')

    def test_source_offsets_and_word_budget_preserve_original_text(self):
        text='🙂 A claim is quoted.\n'+' '.join('word'+str(i) for i in range(120))+'.'
        chunks=units(text)
        raw=text.encode('utf-16-le')
        for row in chunks:
            self.assertEqual(raw[row['start']*2:row['end']*2].decode('utf-16-le'),row['text'])
        evidence,preview=extract_evidence(chunks,[{'evidence_id':2}])
        self.assertLessEqual(len(preview.split()),80)
        self.assertTrue(all(row['text'] in text for row in evidence))

    def test_entailment_filter_drops_unsupported_and_unmeasured_candidates(self):
        from semeval.official_nli import filter_supported
        data=taxonomy()
        parent='URW: Blaming the war on others rather than the invader'
        label=parent+': The West are the aggressors'
        rows=[{'label':label,'parent':parent,'score':.6,'evidence_id':0},
              {'label':parent+': Other','parent':parent,'score':.7,'evidence_id':0}]
        matches,ranking=filter_supported(data,rows,[{'entailment':.9,'contradiction':.05},{'entailment':.95,'contradiction':.03}])
        self.assertEqual([row['label'] for row in matches],[label])
        matches,ranking=filter_supported(data,rows,[None,{'entailment':.1,'contradiction':.5}])
        self.assertEqual(matches[0]['label'],'Other')
        self.assertIsNone(ranking[-1]['support_score'])
