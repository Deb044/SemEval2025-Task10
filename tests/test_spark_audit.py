"""Opt-in Spark SQL integration tests: RUN_SPARK_TESTS=1 python -m unittest tests.test_spark_audit -v."""
import os
import tempfile
import unittest
from pathlib import Path

from semeval.data import write_jsonl


@unittest.skipUnless(os.environ.get('RUN_SPARK_TESTS') == '1', 'Set RUN_SPARK_TESTS=1 for local Java/Spark integration tests')
class SparkAuditChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import pyspark
        from pyspark.sql import SparkSession
        os.environ.setdefault('SPARK_HOME', str(Path(pyspark.__file__).parent))
        os.environ.setdefault('SPARK_LOCAL_IP', '127.0.0.1')
        cls.spark = (SparkSession.builder.master('local[2]').appName('NarrativeLab-AuditTests')
                     .config('spark.ui.enabled', 'false').config('spark.driver.bindAddress', '127.0.0.1')
                     .config('spark.sql.shuffle.partitions', '2').getOrCreate())
        cls.spark.sparkContext.setLogLevel('ERROR')

    @classmethod
    def tearDownClass(cls):
        cls.spark.stop()

    def test_aggregates_preserve_multilingual_inputs_without_exporting_text(self):
        import json
        from experiments.spark_audit import audit
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'test.jsonl'
            write_jsonl(path, [dict(article_id='fixture-en', task=2, language='en', text='PRIVATE_TEST_SENTINEL', labels=['claim']),
                               dict(article_id='fixture-ar', task=2, language='ar', text='نص تجريبي', labels=[])])
            before = path.read_bytes()
            result = audit(self.spark, {'fixture/test': path})
            self.assertEqual(result['records'], 2)
            self.assertEqual({r['language'] for r in result['counts']}, {'en', 'ar'})
            self.assertEqual(result['label_counts'][0]['records'], 1)
            self.assertEqual(path.read_bytes(), before)
            self.assertNotIn('PRIVATE_TEST_SENTINEL', json.dumps(result))
            self.assertNotIn('fixture-en', json.dumps(result))

    def test_rejects_normalized_text_leakage_and_invalid_records(self):
        from experiments.spark_audit import audit
        with tempfile.TemporaryDirectory() as directory:
            train, test = Path(directory) / 'train.jsonl', Path(directory) / 'test.jsonl'
            row = dict(article_id='train', task=2, language='en', text='same\u00a0 words', labels=['claim'])
            write_jsonl(train, [row])
            write_jsonl(test, [dict(row, article_id='test', text='same words')])
            with self.assertRaisesRegex(ValueError, 'overlaps='):
                audit(self.spark, {'fixture/train': train, 'fixture/test': test})
            write_jsonl(test, [dict(row, article_id='test', text=None)])
            with self.assertRaisesRegex(ValueError, 'invalid=1'):
                audit(self.spark, {'fixture/test': test})
