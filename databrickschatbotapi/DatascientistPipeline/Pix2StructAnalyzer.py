from transformers import Pix2StructProcessor, Pix2StructForConditionalGeneration
from PIL import Image

class Pix2StructAnalyzer:
    def __init__(self, model_name='google/matcha-chart2text-statista'):
        self.processor = Pix2StructProcessor.from_pretrained(model_name)
        self.model = Pix2StructForConditionalGeneration.from_pretrained(model_name)

    def generate_description(self, image_path, text_prompt="What does this graph represent", max_tokens=512):
        # Load image
        image = Image.open(image_path)

        # Process inputs
        inputs = self.processor(images=image, text=text_prompt, return_tensors="pt")

        # Generate predictions
        predictions = self.model.generate(**inputs, max_new_tokens=max_tokens)

        # Decode and return the generated description
        generated_description = self.processor.decode(predictions[0], skip_special_tokens=True)
        return generated_description