from transformers import CLIPProcessor, CLIPModel
import torch

from transformers import logging
logging.set_verbosity_error()



def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    model.to(device)
    model.eval()

    return model, processor, device