import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import random
from dotenv import load_dotenv
import os
from huggingface_hub import login
from transformers import ViTModel, ViTFeatureExtractor, BertTokenizer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from data import build_all_dataloaders, batch_convert_and_resize

load_dotenv()
# Setup
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
login(token=HUGGINGFACE_TOKEN)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Hyperparameters
d_model = 768
num_encoder_layers = 6
num_decoder_layers = 6
nhead = 8
max_question_length = 32
max_answer_length = 32
batch_size = 128
num_epochs = 25
learning_rate = 2e-5

# Load models
vit_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
vocab_size = tokenizer.vocab_size

def compute_bleu(reference_tokens, candidate_tokens):
    smoothie = SmoothingFunction().method4
    return sentence_bleu([reference_tokens], candidate_tokens, smoothing_function=smoothie)

def lcs_length(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a)):
        for j in range(len(b)):
            if a[i] == b[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    return dp[len(a)][len(b)]

def compute_rouge_l(reference, candidate):
    ref_tokens = reference.split()
    cand_tokens = candidate.split()
    if len(ref_tokens) == 0 or len(cand_tokens) == 0:
        return 0.0
    lcs = lcs_length(ref_tokens, cand_tokens)
    precision = lcs / len(cand_tokens)
    recall = lcs / len(ref_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)

class MedVQAModel(nn.Module):
    def __init__(self, vocab_size, d_model, num_encoder_layers, num_decoder_layers, nhead,
                 max_question_length, max_answer_length):
        super(MedVQAModel, self).__init__()
        self.d_model = d_model
        self.max_question_length = max_question_length
        self.max_answer_length = max_answer_length

        # Components
        self.image_encoder = ViTModel.from_pretrained("google/vit-base-patch16-224")
        self.image_proj = nn.Linear(self.image_encoder.config.hidden_size, d_model)
        self.text_embedding = nn.Embedding(vocab_size, d_model)
        self.question_pos_embedding = nn.Parameter(torch.zeros(max_question_length, d_model))
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.question_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(d_model=d_model, nhead=nhead)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.answer_embedding = nn.Embedding(vocab_size, d_model)
        self.answer_pos_embedding = nn.Parameter(torch.zeros(max_answer_length, d_model))
        self.generator = nn.Linear(d_model, vocab_size)

        # Initialize positional embeddings
        nn.init.normal_(self.question_pos_embedding, mean=0, std=0.02)
        nn.init.normal_(self.answer_pos_embedding, mean=0, std=0.02)

    def forward(self, pixel_values, question_input_ids, answer_input_ids):
        B = question_input_ids.size(0)
        # Image encoding
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_features = self.image_proj(image_outputs.last_hidden_state)

        # Question encoding
        q_embeds = self.text_embedding(question_input_ids) 
        q_len = q_embeds.size(1)
        pos_q = self.question_pos_embedding[:q_len, :].unsqueeze(0).expand(B, -1, -1)
        q_embeds = q_embeds + pos_q
        q_embeds = q_embeds.transpose(0, 1)
        q_encoded = self.question_encoder(q_embeds)
        q_encoded = q_encoded.transpose(0, 1)

        # Joint encoder output
        encoder_out = torch.cat([image_features, q_encoded], dim=1)
        encoder_out = encoder_out.transpose(0, 1)

        # Answer decoding
        a_embeds = self.answer_embedding(answer_input_ids)
        a_len = a_embeds.size(1)
        pos_a = self.answer_pos_embedding[:a_len, :].unsqueeze(0).expand(B, -1, -1)
        a_embeds = a_embeds + pos_a
        a_embeds = a_embeds.transpose(0, 1)

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(a_embeds.size(0)).to(a_embeds.device)
        decoder_out = self.decoder(tgt=a_embeds, memory=encoder_out, tgt_mask=tgt_mask)
        decoder_out = decoder_out.transpose(0, 1)
        logits = self.generator(decoder_out)
        return logits

    def generate(self, pixel_values, question_input_ids, max_length=32,
                 start_token_id=tokenizer.cls_token_id, end_token_id=tokenizer.sep_token_id):
        self.eval()
        with torch.no_grad():
            B = question_input_ids.size(0)
            # Process image and question
            image_outputs = self.image_encoder(pixel_values=pixel_values)
            image_features = self.image_proj(image_outputs.last_hidden_state)
            q_embeds = self.text_embedding(question_input_ids)
            q_len = q_embeds.size(1)
            pos_q = self.question_pos_embedding[:q_len, :].unsqueeze(0).expand(B, -1, -1)
            q_embeds = q_embeds + pos_q
            q_embeds = q_embeds.transpose(0, 1)
            q_encoded = self.question_encoder(q_embeds)
            q_encoded = q_encoded.transpose(0, 1)
            encoder_out = torch.cat([image_features, q_encoded], dim=1)
            encoder_out = encoder_out.transpose(0, 1)

            # Autoregressive generation
            generated = torch.full((B, 1), start_token_id, dtype=torch.long, device=encoder_out.device)
            for t in range(max_length):
                a_embeds = self.answer_embedding(generated)
                cur_len = a_embeds.size(1)
                pos_a = self.answer_pos_embedding[:cur_len, :].unsqueeze(0).expand(B, -1, -1)
                a_embeds = a_embeds + pos_a
                a_embeds = a_embeds.transpose(0, 1)

                tgt_mask = nn.Transformer.generate_square_subsequent_mask(a_embeds.size(0)).to(a_embeds.device)
                decoder_out = self.decoder(tgt=a_embeds, memory=encoder_out, tgt_mask=tgt_mask)
                decoder_out = decoder_out.transpose(0, 1)
                logits = self.generator(decoder_out)
                next_token_logits = logits[:, -1, :]
                next_tokens = next_token_logits.argmax(dim=-1).unsqueeze(1)
                generated = torch.cat([generated, next_tokens], dim=1)
                if (next_tokens == end_token_id).all():
                    break
            return generated

def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable

def compute_loss_and_accuracy_per_sample(logits, target_input_ids, pad_token_id=tokenizer.pad_token_id):
    B, L, V = logits.shape
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id, reduction="none")
    logits_flat = logits.contiguous().view(B * L, V)
    target_flat = target_input_ids.contiguous().view(B * L)
    loss_all = loss_fn(logits_flat, target_flat).view(B, L)

    losses = []
    accuracies = []
    for i in range(B):
        target_i = target_input_ids[i]
        mask = (target_i != pad_token_id)
        if mask.sum().item() > 0:
            loss_i = loss_all[i][mask].mean()
        else:
            loss_i = torch.tensor(0.0, device=logits.device)
        losses.append(loss_i.item())

        preds_i = logits[i].argmax(dim=-1)
        correct = ((preds_i == target_i) & mask).sum().item()
        total = mask.sum().item()
        acc_i = correct / total if total > 0 else 0.0
        accuracies.append(acc_i)
    return losses, accuracies

def compute_generation_metrics(model, batch, device):
    open_indices = [i for i, qt in enumerate(batch["q_type"]) if qt == "open"]
    if len(open_indices) == 0:
        return [], []

    image_inputs = vit_feature_extractor(images=batch['image'], return_tensors="pt", do_rescale=False)
    q_inputs = tokenizer(
        batch['question'],
        padding="max_length",
        truncation=True,
        max_length=max_question_length,
        return_tensors="pt"
    )

    pixel_values = image_inputs["pixel_values"][open_indices].to(device)
    question_input_ids = q_inputs["input_ids"][open_indices].to(device)
    
    gen_ids = model.generate(pixel_values, question_input_ids, max_length=max_answer_length)
    generated_texts = [tokenizer.decode(gen_id, skip_special_tokens=True) for gen_id in gen_ids]
    references = [batch["raw_answers"][i] for i in open_indices]

    bleu_scores = []
    rouge_scores = []
    for ref, cand in zip(references, generated_texts):
        bleu = compute_bleu(ref.split(), cand.split())
        rouge = compute_rouge_l(ref, cand)
        bleu_scores.append(bleu)
        rouge_scores.append(rouge)
    return bleu_scores, rouge_scores

def compute_loss_and_accuracy(logits, target_input_ids, pad_token_id=tokenizer.pad_token_id):
    logits = logits.contiguous().view(-1, logits.size(-1))
    target = target_input_ids.contiguous().view(-1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=pad_token_id)
    loss = loss_fn(logits, target)
    preds = logits.argmax(dim=-1)
    non_pad = target != pad_token_id
    correct = ((preds == target) & non_pad).sum().item()
    total = non_pad.sum().item() if non_pad.sum().item() > 0 else 1
    acc = correct / total
    return loss, acc

def train_epoch(model, dataloader, optimizer, device):
    model.train()
    # Initialize trackers
    total_loss_overall = total_acc_overall = count_overall = 0
    total_loss_open = total_acc_open = count_open = 0
    total_loss_closed = total_acc_closed = count_closed = 0
    bleu_sum = rouge_sum = gen_open_count = 0

    for batch in tqdm(dataloader, desc="Training Epoch"):
        optimizer.zero_grad()
        # Process inputs
        image_inputs = vit_feature_extractor(images=batch['image'], return_tensors="pt", do_rescale=False)
        q_inputs = tokenizer(
            batch['question'],
            padding="max_length",
            truncation=True,
            max_length=max_question_length,
            return_tensors="pt"
        )
        a_inputs = tokenizer(
            batch['answer'],
            padding="max_length",
            truncation=True,
            max_length=max_answer_length,
            return_tensors="pt"
        )

        # To device
        pixel_values = image_inputs["pixel_values"].to(device)
        question_input_ids = q_inputs["input_ids"].to(device)
        answer_input_ids = a_inputs["input_ids"].to(device)

        # Forward and loss
        logits = model(pixel_values, question_input_ids, answer_input_ids[:, :-1])
        losses, accs = compute_loss_and_accuracy_per_sample(logits, answer_input_ids[:, 1:])
        batch_q_types = batch["q_type"]
        
        # Track metrics by question type
        for i, qtype in enumerate(batch_q_types):
            total_loss_overall += losses[i]
            total_acc_overall += accs[i]
            count_overall += 1
            if qtype == "open":
                total_loss_open += losses[i]
                total_acc_open += accs[i]
                count_open += 1
            elif qtype == "closed":
                total_loss_closed += losses[i]
                total_acc_closed += accs[i]
                count_closed += 1

        # Backprop
        overall_loss, _ = compute_loss_and_accuracy(logits, answer_input_ids[:, 1:])
        overall_loss.backward()
        optimizer.step()

        # Generation metrics
        model.eval()
        with torch.no_grad():
            bleu_scores, rouge_scores = compute_generation_metrics(model, batch, device)
        model.train()
        if bleu_scores:
            bleu_sum += sum(bleu_scores)
            rouge_sum += sum(rouge_scores)
            gen_open_count += len(bleu_scores)

    # Calculate averages
    overall_loss_avg = total_loss_overall / count_overall if count_overall > 0 else 0
    overall_acc_avg = total_acc_overall / count_overall if count_overall > 0 else 0
    open_loss_avg = total_loss_open / count_open if count_open > 0 else 0
    open_acc_avg = total_acc_open / count_open if count_open > 0 else 0
    closed_loss_avg = total_loss_closed / count_closed if count_closed > 0 else 0
    closed_acc_avg = total_acc_closed / count_closed if count_closed > 0 else 0
    open_bleu_avg = bleu_sum / gen_open_count if gen_open_count > 0 else 0
    open_rouge_avg = rouge_sum / gen_open_count if gen_open_count > 0 else 0

    # Log metrics
    print("Training Epoch Metrics:")
    print(f" Overall Loss: {overall_loss_avg:.4f} | Overall Accuracy: {overall_acc_avg:.4f}")
    print(f" Closed Q - Loss: {closed_loss_avg:.4f} | Acc: {closed_acc_avg:.4f}")
    print(f" Open Q   - Loss: {open_loss_avg:.4f} | Acc: {open_acc_avg:.4f}")
    print(f" Open-ended BLEU: {open_bleu_avg:.4f} | ROUGE-L: {open_rouge_avg:.4f}")

    return (overall_loss_avg, overall_acc_avg, closed_loss_avg, closed_acc_avg,
            open_loss_avg, open_acc_avg, open_bleu_avg, open_rouge_avg)

def evaluate_epoch(model, dataloader, device):
    model.eval()
    # Initialize trackers
    total_loss_overall = total_acc_overall = count_overall = 0
    total_loss_open = total_acc_open = count_open = 0
    total_loss_closed = total_acc_closed = count_closed = 0
    bleu_sum = rouge_sum = gen_open_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating Epoch"):
            # Process inputs
            image_inputs = vit_feature_extractor(images=batch['image'], return_tensors="pt", do_rescale=False)
            q_inputs = tokenizer(
                batch['question'],
                padding="max_length",
                truncation=True,
                max_length=max_question_length,
                return_tensors="pt"
            )
            a_inputs = tokenizer(
                batch['answer'],
                padding="max_length",
                truncation=True,
                max_length=max_answer_length,
                return_tensors="pt"
            )

            # To device
            pixel_values = image_inputs["pixel_values"].to(device)
            question_input_ids = q_inputs["input_ids"].to(device)
            answer_input_ids = a_inputs["input_ids"].to(device)

            # Forward and metrics
            logits = model(pixel_values, question_input_ids, answer_input_ids[:, :-1])
            losses, accs = compute_loss_and_accuracy_per_sample(logits, answer_input_ids[:, 1:])
            batch_q_types = batch["q_type"]
            
            # Track metrics by question type
            for i, qtype in enumerate(batch_q_types):
                total_loss_overall += losses[i]
                total_acc_overall += accs[i]
                count_overall += 1
                if qtype == "open":
                    total_loss_open += losses[i]
                    total_acc_open += accs[i]
                    count_open += 1
                elif qtype == "closed":
                    total_loss_closed += losses[i]
                    total_acc_closed += accs[i]
                    count_closed += 1

            # Generation metrics
            bleu_scores, rouge_scores = compute_generation_metrics(model, batch, device)
            if bleu_scores:
                bleu_sum += sum(bleu_scores)
                rouge_sum += sum(rouge_scores)
                gen_open_count += len(bleu_scores)

    # Calculate averages
    overall_loss_avg = total_loss_overall / count_overall if count_overall > 0 else 0
    overall_acc_avg = total_acc_overall / count_overall if count_overall > 0 else 0
    open_loss_avg = total_loss_open / count_open if count_open > 0 else 0
    open_acc_avg = total_acc_open / count_open if count_open > 0 else 0
    closed_loss_avg = total_loss_closed / count_closed if count_closed > 0 else 0
    closed_acc_avg = total_acc_closed / count_closed if count_closed > 0 else 0
    open_bleu_avg = bleu_sum / gen_open_count if gen_open_count > 0 else 0
    open_rouge_avg = rouge_sum / gen_open_count if gen_open_count > 0 else 0

    # Log metrics
    print("Evaluation Epoch Metrics:")
    print(f" Overall Loss: {overall_loss_avg:.4f} | Overall Accuracy: {overall_acc_avg:.4f}")
    print(f" Closed Q - Loss: {closed_loss_avg:.4f} | Acc: {closed_acc_avg:.4f}")
    print(f" Open Q   - Loss: {open_loss_avg:.4f} | Acc: {open_acc_avg:.4f}")
    print(f" Open-ended BLEU: {open_bleu_avg:.4f} | ROUGE-L: {open_rouge_avg:.4f}")

    return (overall_loss_avg, overall_acc_avg, closed_loss_avg, closed_acc_avg,
            open_loss_avg, open_acc_avg, open_bleu_avg, open_rouge_avg)

def sample_and_log(model, dataloader, device, num_samples=5):
    # Sample predictions for visualization
    model.eval()
    dataset_list = list(dataloader.dataset)
    samples = random.sample(dataset_list, min(num_samples, len(dataset_list)))
    
    print("\nSampling and logging model predictions:")
    with torch.no_grad():
        for idx, sample in enumerate(samples):
            print(f"\n--- Sample {idx+1} ---")
            question = sample["question"]
            true_answer = sample["answer"]
            q_type = "closed" if true_answer.strip().lower() in ["yes", "no"] else "open"

            # Generate answer
            image_tensor = sample["image"].unsqueeze(0).to(device)
            image_input = vit_feature_extractor(image_tensor, return_tensors="pt", do_rescale=False)["pixel_values"].to(device)
            q_input = tokenizer(
                question,
                padding="max_length",
                truncation=True,
                max_length=max_question_length,
                return_tensors="pt"
            )["input_ids"].to(device)

            gen_ids = model.generate(image_input, q_input, max_length=max_answer_length)
            gen_answer = tokenizer.decode(gen_ids[0], skip_special_tokens=True)

            print("Question      :", question)
            print("True Answer   :", true_answer)
            print("Generated Ans :", gen_answer)
            print("Question Type :", q_type)
