import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTFeatureExtractor, BertTokenizer
from model import MedVQAModel  
import argparse
import json
import os
import sys

class MedVQAInference:
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.d_model = 768
        self.num_encoder_layers = 6
        self.num_decoder_layers = 6
        self.nhead = 8
        self.max_question_length = 32
        self.max_answer_length = 32
        
        if self.verbose:
            print("Loading feature extractor and tokenizer...", file=sys.stderr)
        self.vit_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.vocab_size = self.tokenizer.vocab_size
        if self.verbose:
            print("Feature extractor and tokenizer loaded.\n", file=sys.stderr)


        self.model = MedVQAModel(
            vocab_size=self.vocab_size,
            d_model=self.d_model,
            num_encoder_layers=self.num_encoder_layers,
            num_decoder_layers=self.num_decoder_layers,
            nhead=self.nhead,
            max_question_length=self.max_question_length,
            max_answer_length=self.max_answer_length
        )

        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        if self.verbose:
            print("Current working directory:", os.getcwd(), file=sys.stderr)
            print("Loading model...", file=sys.stderr)

        model_path = "/Users/tousif/Desktop/IITK/Academics /Sem 6/CS776/Project/Final Code/server/src/models/pathology/final_med_vqa_path_model.pt"
        state_dict = torch.load(model_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)
        if self.verbose:
            print("Model loaded successfully", file=sys.stderr)

    def generate_answer(self, image_path, question):
        """
        Given an image path and a question, preprocess the image and question,
        then use the model to generate and decode an answer.
        """
        self.model.eval()
        image = Image.open(image_path)
        transformed_image = self.transform(image).unsqueeze(0)  

        image_inputs = self.vit_feature_extractor(
            images=transformed_image, 
            return_tensors="pt", 
            do_rescale=False
        )

        q_inputs = self.tokenizer(
            question,
            padding="max_length",
            truncation=True,
            max_length=self.max_question_length,
            return_tensors="pt"
        )
        
        pixel_values = image_inputs["pixel_values"]
        question_input_ids = q_inputs["input_ids"]

        gen_ids = self.model.generate(pixel_values, question_input_ids, max_length=self.max_answer_length)
        gen_answer = self.tokenizer.decode(gen_ids[0], skip_special_tokens=True)
        
        return gen_answer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process pathology images for closed-ended VQA')
    parser.add_argument('--image', required=True, help='Path to the image file')
    parser.add_argument('--question', required=True, help='Question about the image')

    args = parser.parse_args()

    model = MedVQAInference()
    answer = model.generate_answer(args.image, args.question)
    
    result = {
        "success": True,
        "answer": answer
    }
    
    print(json.dumps(result))