import spacy
from spacy.tokens import Token, Span
import logging
from flashtext import KeywordProcessor


class FlashPatternMatcher(object):
    name = "flashpatternmatcher"  # component name shown in pipeline

    def __init__(self, nlp, patterns, patterns_by_class, default_label=None):
        """
        Initialise the Spacy pipeline component

        Set up the extensions on the Tokens and Spans.


        :param nlp: Spacy NLP engine
        :param patterns: List of dicts of patterns to match on
        :param patterns_by_class: List of dicts of patterns to match on, grouped by entity type
        :param default_label: default label to use on matched entities.
        """
        self.nlp = nlp
        if default_label is None:
            self.default_label = "CUSTOM"
        else:
            self.default_label = default_label
        _ = self.nlp.tokenizer.vocab[self.default_label]  # add string to vocab
        self.nlp.get_pipe("ner").add_label(self.default_label)  # add string to vocab
        self.patterns = patterns
        self.patterns_by_class = patterns_by_class
        # initialise the matcher and add patterns
        self.keyword_processor = KeywordProcessor()
        for k, v in self.patterns_by_class.items():
            _ = self.nlp.tokenizer.vocab[k]  # add string to vocab
            self.nlp.get_pipe("ner").add_label(k)  # add string to vocab
            self.keyword_processor.add_keywords_from_list(self.patterns_by_class[k])
        try:
            Token.set_extension("original_label", default=None)
        except ValueError:  # do not force overwrite if extension already set
            pass
        try:
            Span.set_extension(
                "original_label",
                getter=lambda span: list(
                    set([token._.original_label for token in span])
                ),
            )
        except ValueError:  # do not force overwrite if extension already set
            pass
        # no callback function on the matcher patterns.
        logging.debug("PMC Flashget based pattern matcher added.")

    def __call__(self, doc):
        """
        Called when the pipeline step runs. Properties added here so they are added pre-merge.

        :param doc: Spacy's doc object
        :return: Spacy doc object (required so that the doc can be passed to the next pipeline step)
        """
        for ent in doc.ents:  # set an extension property on each token showing the original ent label
            for token in ent:
                token._.set('original_label', ent.label)
        matches = self.keyword_processor.extract_keywords(doc.text, span_info=True)
        spans = []  # keep the spans for later so we can merge them afterwards
        for _, start, end in matches:
            label = self.patterns[doc.char_span(start, end).text][0]["ent_type"]
            if label is not None:
                entity = doc.char_span(start, end, label=label)
            else:
                entity = doc.char_span(start, end, label=self.default_label)
            spans.append(entity)
        doc.ents = list(doc.ents) + spans  # overwrite doc.ents and add custom entities
        for span in spans:
            span.merge()  # merge all spans at the end to avoid mismatched indices
        return doc  # return the doc so that the Pipeline can work.


def initialise_nlp(lang_model=None, patterns=None, classes=None, label=None):
    """

    Initialise Spacy, using patterns and classes.

    If no vocab is found, Spacy will fallback to initialise the pipeline with no custom vocab processing,
    but will continue to filter the entities by the allowed entity types.

    :param lang_model: Spacy language model to load.
    :param patterns:
    :param classes:
    :param label: default label to add to custom entities.
    :return: Spacy NLP
    """
    if lang_model is None:
        lang_model = "en_core_web_sm"
    nlp = spacy.load(lang_model)
    logging.info("Loaded basic Spacy natural language model")
    if patterns and classes:
        logging.info("Parsed vocabulary data")
        nlp.add_pipe(
            FlashPatternMatcher(
                nlp, patterns_by_class=classes, patterns=patterns, default_label=label
            ),
            after="ner",
        )
    return nlp
