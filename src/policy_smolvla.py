import torch
from PIL import Image
from .utils import decode_action


class SmolVLAPolicy:

    def __init__(self, model, processor):

        self.model = model
        self.processor = processor

    def predict(self, obs, instruction):

        image = Image.fromarray(obs["agentview_image"])

        inputs = self.processor(
            images=image,
            text=instruction,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():

            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10
            )

        action_text = self.processor.decode(
            outputs[0],
            skip_special_tokens=True
        )

        action = decode_action(action_text)

        return action