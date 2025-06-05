import sentencepiece as spm

class SentencePieceTokenizer:
    def __init__(self, model_prefix=None, vocab_size=32000, model_type='bpe', character_coverage=1.0):
        self.model_prefix = model_prefix
        self.vocab_size = vocab_size
        self.model_type = model_type
        self.character_coverage = character_coverage
        self.sp = None

    def export_text_file(self, df, filename, col):
        with open(filename, 'w', encoding='utf-8') as f:
            for line in df[col]:
                f.write(str(line).strip() + '\n')

    def train(self, input_file):
        spm.SentencePieceTrainer.Train(
            input=input_file,
            model_prefix=self.model_prefix,
            vocab_size=self.vocab_size,
            character_coverage=self.character_coverage,
            model_type=self.model_type
        )

    def load(self, model_file=None):
        if model_file is None:
            if self.model_prefix is None:
                raise ValueError("model_file or model_prefix must be provided to load model.")
            model_file = f"{self.model_prefix}.model"
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_file)

    def encode(self, sentences):
        if self.sp is None:
            raise ValueError("Model not loaded. Call load() first.")
        if isinstance(sentences, str):
            return self.sp.encode_as_pieces(sentences)
        elif isinstance(sentences, list):
            return [self.sp.encode_as_pieces(s) for s in sentences]
        else:
            raise TypeError("Input must be str or list of str.")

    def save_tokenized(self, encoded_sentences, filename):
        with open(filename, 'w', encoding='utf-8') as f:
            for tokens in encoded_sentences:
                f.write(' '.join(tokens) + '\n')
