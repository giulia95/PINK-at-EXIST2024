import models
MODELS = {
    'pink_mlp': models.PinkMLP,
    'pink_transformer': models.PinkTransformer,
}

MODALITY_ENCODERS = {
    'clip_caption': models.LinearModalityEncoder,
    'clip_image': models.LinearModalityEncoder,
    'clip_text': models.LinearModalityEncoder,
    'bert_caption': models.LinearModalityEncoder,
    'bert_text': models.LinearModalityEncoder,
}
