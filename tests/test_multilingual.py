"""Native-language protocol checks using invented fixtures, not benchmark data."""
import unittest
from experiments.public_multilingual import GROUPS,grouped_splits,is_portugal,record

class MultilingualChecks(unittest.TestCase):
    def test_hierarchy_has_native_25_labels_and_six_parents(self):
        self.assertEqual(len(GROUPS),6)
        labels=[label for group in GROUPS.values() for label in group]
        self.assertEqual(len(labels),25);self.assertEqual(len(set(labels)),25)
        self.assertIn('Appeal_to_Pity',GROUPS['Distraction'])
        self.assertIn('False_Equivalence',GROUPS['Simplification'])

    def test_portugal_filter_rejects_brazilian_and_unknown_publishers(self):
        self.assertTrue(is_portugal(dict(language='PT',media_name='publico.pt')))
        self.assertFalse(is_portugal(dict(language='PT',media_name='uol.com.br')))
        self.assertFalse(is_portugal(dict(language='PT',media_name='unknown.pt')))
        self.assertFalse(is_portugal(dict(language='ITA',media_name='publico.pt')))

    def test_grouping_joins_duplicates_and_removes_conflicting_labels(self):
        rows=[record('fixture','bg',str(i),f'Invented source text number {i} with enough words',['A'],f'doc-{i//2}') for i in range(40)]
        rows += [record('fixture','bg','copy',rows[0]['text'],['A'],'doc-8'),
                 record('fixture','bg','conflict',rows[5]['text'],['B'],'doc-12')]
        parts,counts=grouped_splits(rows)
        self.assertEqual(counts['conflicting_duplicate_records'],2)
        self.assertEqual(counts['collapsed_duplicates'],1)
        groups=[{r['duplicate_group'] for r in part} for part in parts.values()]
        self.assertFalse(any(groups[i]&groups[j] for i in range(3) for j in range(i)))
        texts=[r['text'] for part in parts.values() for r in part]
        self.assertEqual(len(texts),len(set(texts)))
        self.assertNotIn(rows[5]['text'],texts)
