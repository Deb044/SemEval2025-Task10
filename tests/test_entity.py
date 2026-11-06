"""Entity framing invariants. Invented fixtures do not establish task accuracy."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import hashlib
import numpy as np
from semeval import entity


def row(text,name,role,identity):
    start=text.index(name)
    return dict(task=1,article_id=identity,language='en',text=text,mention=name,start=start,end=start+len(name),end_exclusive=start+len(name),labels=[role],main_role=entity.ROLE_PARENT[role],annotation_source='invented test fixture')

class Encoder:
    def encode(self,texts):
        values=np.array([list(hashlib.sha256(t.encode()).digest()[:16]) for t in texts],dtype='float32')/255
        return values/np.linalg.norm(values,axis=1,keepdims=True)

class EntityChecks(unittest.TestCase):
    def test_exact_mention_units_and_surrogate_boundaries(self):
        text='😀 Mira helped Oleg.'
        rows=entity.mentions_from_payload(dict(text=text,language='en',offset_unit='utf16',mentions=[dict(mention='Mira',start=3,end=7)]))
        self.assertEqual(rows[0]['start'],2)
        self.assertEqual(rows[0]['text'],text)
        with self.assertRaises(ValueError):entity.utf16_index(text,1)
        with self.assertRaises(ValueError):entity.mentions_from_payload(dict(text=text,mentions=[dict(mention='Mira',start=0,end=4)]))

    def test_repeated_mentions_keep_the_selected_occurrence(self):
        text='Mira accused Oleg. Later Mira defended Oleg.'
        start=text.rindex('Mira');r=dict(task=1,article_id='a',language='en',text=text,mention='Mira',start=start,end=start+4,end_exclusive=start+4)
        marked=entity.context(r)['marked']
        self.assertIn('Later  [TARGET] Mira [/TARGET]  defended',marked)
        self.assertEqual(marked.count('[TARGET]'),1)

    def test_entity_views_distinguish_targets_in_the_same_article(self):
        text='Mira protected Nila while Oleg misled the public.'
        first=row(text,'Mira','Guardian','one');second=row(text,'Oleg','Deceiver','one')
        values=entity.representations([first,second],Encoder())
        self.assertEqual(values.shape,(2,48))
        np.testing.assert_allclose(values[0,-16:],values[1,-16:])
        self.assertFalse(np.array_equal(values[0,:32],values[1,:32]))

    def test_role_decoder_keeps_one_parent_and_multiple_children(self):
        fine=np.full((1,22),.01);parent=np.array([[.01,.01,.98]])
        for role in ('Guardian','Virtuous'):fine[0,entity.LABELS.index(role)]=.9
        fine[0,entity.LABELS.index('Tyrant')]=1
        value=entity.decode(fine,parent,.2)[0]
        self.assertEqual(value['main_role'],'Protagonist')
        self.assertEqual(set(value['labels']),{'Guardian','Virtuous'})
        abstained=entity.decode(np.zeros((1,22)),parent,.2,True)[0]
        self.assertTrue(abstained['abstained']);self.assertEqual(abstained['labels'],[])

    def test_training_prediction_and_official_export_roundtrip(self):
        train=[row(f'Example {i}: Mira is portrayed as {role}.','Mira',role,'train'+str(i)) for i,role in enumerate(['Guardian','Deceiver','Victim']*3)]
        dev=[row(f'Development {i}: Oleg appears as {role}.','Oleg',role,'dev'+str(i)) for i,role in enumerate(['Guardian','Deceiver','Victim'])]
        with patch('semeval.features.get_encoder',return_value=Encoder()):
            model=entity.train(train,dev)
            predictions=entity.predict(model,dev)
        scores=entity.evaluate(dev,predictions)
        self.assertEqual(scores['metrics']['count'],3)
        self.assertEqual(len(scores['metrics']['per_label']),22)
        self.assertEqual(model['head'].initial.shape[0],25)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'roles.tsv';entity.export_tsv(predictions,path)
            self.assertEqual(len(path.read_text(encoding='utf-8').splitlines()),3)
            invalid=dict(predictions[0],main_role='Other')
            with self.assertRaises(ValueError):entity.export_tsv([invalid],path)
        bad=[dict(r,label_schema='native') for r in train]
        with self.assertRaises(ValueError):entity.train(bad,dev)
