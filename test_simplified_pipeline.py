import unittest
import spacy
from simplified_pipeline import initialise_nlp


# noinspection PyUnresolvedReferences
class Pipeline(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        patterns = {
            'John Smith': [
                {
                    'ent_label': 'John Smith',
                    'ent_type': 'ARTIST',
                    'ent_action': 'Replace',
                    'ent_data': None,
                }
            ],
            'Mary Sanchez': [
                {
                    'ent_label': 'Mary Sanchez',
                    'ent_type': None,
                    'ent_action': 'Add',
                    'ent_data': None,
                }
            ],
        }
        classes = {'Artist': ['John Smith', 'Mary Sanchez']}
        cls.nlp = initialise_nlp(patterns=patterns, classes=classes, label='CUSTOM')
        Pipeline.doc = cls.nlp(
            'John Smith was a famous painter, who regularly exhibited with Mary Sanchez, \
                                    the engraver.'
        )

    @classmethod
    def tearDownClass(cls):
        del cls.nlp
        del cls.doc

    def test_language(self):
        """
        Check for valid Spacy instance, with English languade model.
        """
        self.assertIsInstance(Pipeline.nlp, spacy.lang.en.English)

    def test_pipenames(self):
        """
        Check for Spacy pipeline with valid list of pipeline steps in the correct order
        """
        self.assertEqual(
            Pipeline.nlp.pipe_names, ['tagger', 'parser', 'ner', 'flashpatternmatcher']
        )

    def test_bare_ents(self):
        """
        Check Spacy returns a tuple for the entities in the doc
        """
        self.assertIsInstance(Pipeline.doc.ents, tuple)

    def test_bare_ents_test(self):
        """
        Check the correct text in the doc phrase has been tagged as an entity
        """
        self.assertEqual(Pipeline.doc.ents[0].text, 'John Smith')
        self.assertEqual(Pipeline.doc.ents[1].text, 'Mary Sanchez')

    def test_ents_labels(self):
        """
        Check that labels are right when a custom default label IS set.
        """
        self.assertListEqual([x.label_ for x in Pipeline.doc.ents], ['ARTIST', 'CUSTOM'])

    def test_ents_original_labels(self):
        self.assertListEqual([x._.original_label for x in Pipeline.doc.ents], [[378], [378]])


# noinspection PyUnresolvedReferences
class Pipeline2(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        p = {
            'Foo': [
                {
                    'ent_label': 'Foo',
                    'ent_type': 'TRIBE',
                    'ent_action': 'Replace',
                    'ent_data': None,
                }
            ],
            'F00': [
                {
                    'ent_label': 'Foo',
                    'ent_type': 'TRIBE',
                    'ent_action': 'Replace',
                    'ent_data': None,
                }
            ],
            'Fooo': [
                {
                    'ent_label': 'Foo',
                    'ent_type': 'TRIBE',
                    'ent_action': 'Replace',
                    'ent_data': None,
                }
            ],
            'Bar': [
                {
                    'ent_label': 'Bar',
                    'ent_type': 'GPE',
                    'ent_action': 'Add',
                    'ent_data': None,
                }
            ],
            'Baar': [
                {
                    'ent_label': 'Bar',
                    'ent_type': 'GPE',
                    'ent_action': 'Add',
                    'ent_data': None,
                }
            ],
            'BaRR': [
                {
                    'ent_label': 'Bar',
                    'ent_type': 'GPE',
                    'ent_action': 'Add',
                    'ent_data': None,
                }
            ],
        }
        c = {'TRIBE': ['F00', 'Foo', 'Fooo'], 'GPE': ['Baar', 'Bar', 'BaRR']}
        cls.nlp = initialise_nlp(patterns=p, classes=c)
        cls.doc = cls.nlp(
            'The Foo were a lesser known tribe, who lived in the country of Bar.'
        )

    @classmethod
    def tearDownClass(cls):
        del cls.nlp
        del cls.doc

    def test_language(self):
        """
        Check for valid Spacy instance, with English languade model.
        """
        self.assertIsInstance(Pipeline2.nlp, spacy.lang.en.English)

    def test_pipenames(self):
        """
        Check for Spacy pipeline with valid list of pipeline steps in the correct order
        """
        self.assertEqual(
            Pipeline2.nlp.pipe_names, ['tagger', 'parser', 'ner', 'flashpatternmatcher']
        )

    def test_bare_ents(self):
        """
        Check Spacy returns a tuple for the entities in the doc
        """
        self.assertIsInstance(Pipeline2.doc.ents, tuple)

    def test_bare_ents_test(self):
        """
        Check the correct text in the doc phrase has been tagged as an entity
        """
        self.assertEqual(Pipeline2.doc.ents[0].text, 'Foo')
        self.assertEqual(Pipeline2.doc.ents[1].text, 'Bar')

    def test_ents_labels(self):
        """
        Check that labels are right when the default label is not set.
        """
        self.assertTrue(all([x.label_ != 'CUSTOM' for x in Pipeline2.doc.ents]))

    def test_ents_original_labels(self):
        self.assertListEqual([x._.original_label for x in Pipeline2.doc.ents], [[9191306739292312949], [None]])

