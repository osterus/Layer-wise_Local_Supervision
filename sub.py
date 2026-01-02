import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="text8",
    model_prefix="sp_text8",
    vocab_size=16000,
    model_type="unigram",
    character_coverage=1.0,
    max_sentence_length=1000000000,  # ← ВАЖНО
    bos_id=-1,
    eos_id=-1,
    unk_id=0
)