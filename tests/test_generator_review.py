"""Review aggregation must never manufacture human coverage or average unmatched items."""
import unittest
from experiments.generator_comparison import valid_generation
from experiments.named_review import aggregate
from semeval.review import item_hash


class GeneratorReviewChecks(unittest.TestCase):
    def fixture(self):
        item = {'review_id': 'fixture', 'variant': 'model-a', 'text': 'A speaker made a claim.',
                'target_description': 'Claim', 'explanation': 'The speaker made a claim.'}
        rating = {'review_id': 'fixture', 'reviewer_type': 'llm', 'item_sha256': item_hash(item),
                  'supported': 2, 'attribution': 2, 'relevance': 2, 'readability': 2,
                  'rationale': 'Fixture only.', 'evidence_ids': [0]}
        return item, rating

    def test_no_human_rating_produces_no_combined_score(self):
        item, rating = self.fixture()
        result = aggregate([item], [rating])
        self.assertEqual(result['human_paired_items'], 0)
        self.assertFalse(result['combined_score_available'])
        self.assertIsNone(result['runs'][0]['combined_overall_mean'])
        self.assertEqual(result['runs'][0]['llm_overall_mean'], 2)

    def test_only_bound_genuine_human_scores_can_be_averaged(self):
        item, rating = self.fixture()
        human = dict(rating, reviewer='fixture-person', reviewer_type='human', human_reviewer=True, language_confident=True,
                     supported=0, attribution=0, relevance=0, readability=0)
        result = aggregate([item], [rating], [human])
        self.assertEqual(result['runs'][0]['combined_overall_mean'], 1)
        self.assertEqual(result['human_paired_items'], 1)
        with self.assertRaises(ValueError):
            aggregate([item], [rating], [dict(human, item_sha256='wrong')])
        with self.assertRaises(ValueError):
            aggregate([item], [rating], [dict(human, human_reviewer=False)])

    def test_truncated_or_overlong_generation_is_not_a_valid_explanation(self):
        self.assertTrue(valid_generation({'text': 'A supported claim.', 'finish_reason': 'stop'}))
        self.assertFalse(valid_generation({'text': 'A supported claim.', 'finish_reason': 'length'}))
        self.assertFalse(valid_generation({'text': 'word ' * 81, 'finish_reason': 'stop'}))
