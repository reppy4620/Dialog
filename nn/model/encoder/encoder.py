from transformers import AutoModel


def build_encoder(model_name):
    encoder = AutoModel.from_pretrained(model_name).eval()
    return encoder
