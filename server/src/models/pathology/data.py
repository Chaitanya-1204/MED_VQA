# Import necessary libraries
from datasets import load_dataset, concatenate_datasets
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm
from huggingface_hub import login
from dotenv import load_dotenv
import os 

print(load_dotenv())
# Setup
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
login(token=HUGGINGFACE_TOKEN)


# Define a custom Dataset class for VQA
class VQADataset(Dataset):
    def __init__(self, data):
        """
        Args:
            data (dict): Contains keys "images", "questions", "answers", and "q_type".
        """
        self.images = data["images"]
        self.questions = data["questions"]
        self.answers = data["answers"]
        self.q_type = data["q_type"]
        self.raw_answers = data["raw_answers"]

    def __len__(self):
        return len(self.answers)

    def __getitem__(self, idx):
        return {
            "image": self.images[idx],
            "question": self.questions[idx],
            "answer": self.answers[idx],
            "q_type": self.q_type[idx] , 
            "raw_answers": self.raw_answers[idx]
        }

def split_dataset(dataset):
    """
    Splits a Hugging Face dataset into lists for images, questions, and answers.
    
    Args:
        dataset: A Hugging Face dataset split.
    
    Returns:
        tuple: (images, questions, answers)
    """
    images, questions, answers = [], [], []
    for item in tqdm(dataset, desc="Splitting dataset"):
        images.append(item['image'])
        questions.append(item['question'])
        answers.append(item['answer'])
    return images, questions, answers

def batch_convert_and_resize(images, size=(224, 224)):
    """
    Resizes, converts, and normalizes a list of images.
    
    Args:
        images (list): List of PIL images.
        size (tuple): Desired output size.
    
    Returns:
        torch.Tensor: A tensor of stacked transformed images.
    """
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                                     std=[0.26862954, 0.26130258, 0.27577711])
    transform = transforms.Compose([
    transforms.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
            transforms.Resize((224, 224)),
            transforms.ToTensor()
 
])
    transformed_images = [transform(img) for img in tqdm(images, desc="Transforming images")]
    print(type(transformed_images))
    return transformed_images

def build_open_closed(images, questions, answers):
    """
    Processes answers by stripping whitespace, converting to lowercase, and determining
    question type (closed for "yes"/"no" answers, open otherwise). Images are left unchanged.
    
    Args:
        images (torch.Tensor): Tensor of images.
        questions (list): List of question strings.
        answers (list): List of answer strings.
    
    Returns:
        dict: Contains "images", "questions", "answers", and "q_type".
    """
    processed_answers = []
    raw_answers = []
    q_types = []
    for ans in tqdm(answers, desc="Processing answers"):
        raw_answers.append(ans)
        ans_clean = ans.strip().lower()
        processed_answers.append(ans_clean)
        q_types.append("closed" if ans_clean in ["yes", "no"] else "open")
    return {"images": images, "questions": questions, "answers": processed_answers, "q_type": q_types , "raw_answers": raw_answers}

def build_all_dataloaders(batch_size=16):
    """
    Loads the VQA dataset, combines train and validation splits for training,
    and splits the test split in half to serve as validation and test sets.
    
    Args:
        batch_size (int): Batch size for DataLoaders.
    
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    print("Loading VQA dataset...")
    ds = load_dataset("flaviagiammarino/path-vqa")
    
    # Prepare the combined training set
    print("Combining train and validation splits for training...")
    combined_train_val = concatenate_datasets([ds['train'], ds['validation']])
    print("Splitting combined training data...")
    train_images, train_questions, train_answers = split_dataset(combined_train_val)
    
    print("Transforming training images...")
    train_images_tensor = batch_convert_and_resize(train_images)
    print("Processing training answers and determining question types...")
    train_data = build_open_closed(train_images_tensor, train_questions, train_answers)
    train_loader = DataLoader(VQADataset(train_data), batch_size=batch_size, shuffle=True)
    
    # Split the test set into validation and test parts
    print("Splitting test dataset into test and validation splits...")
    test_images_all, test_questions_all, test_answers_all = split_dataset(ds['test'])
    split_index = len(test_answers_all) // 2
    test_images_part = test_images_all[:split_index]
    test_questions_part = test_questions_all[:split_index]
    test_answers_part = test_answers_all[:split_index]
    val_images_part = test_images_all[split_index:]
    val_questions_part = test_questions_all[split_index:]
    val_answers_part = test_answers_all[split_index:]
    
    print("Transforming test images...")
    test_images_tensor = batch_convert_and_resize(test_images_part)
    print("Transforming validation images...")
    val_images_tensor = batch_convert_and_resize(val_images_part)
    
    print("Processing test answers and determining question types...")
    test_data = build_open_closed(test_images_tensor, test_questions_part, test_answers_part)
    print("Processing validation answers and determining question types...")
    val_data = build_open_closed(val_images_tensor, val_questions_part, val_answers_part)
    
    test_loader = DataLoader(VQADataset(test_data), batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(VQADataset(val_data), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

