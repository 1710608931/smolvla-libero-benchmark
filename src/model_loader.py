import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from transformers import BitsAndBytesConfig


def load_model(model_path, quantization=None):

    processor = AutoProcessor.from_pretrained(model_path)

    if quantization == "int8":

        quant_config = BitsAndBytesConfig(load_in_8bit=True)

        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto"
        )

    elif quantization == "int4":

        quant_config = BitsAndBytesConfig(load_in_4bit=True)

        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            quantization_config=quant_config,
            device_map="auto"
        )

    else:

        model = AutoModelForVision2Seq.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

    model.eval()

    return model, processor