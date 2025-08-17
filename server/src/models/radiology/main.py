import torch
import torch.optim as optim
from data import build_all_dataloaders
from model import MedVQAModel, count_parameters, train_epoch, evaluate_epoch, sample_and_log
from transformers import ViTFeatureExtractor, BertTokenizer

def main():
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    
    # Build DataLoaders 
    print("Building DataLoaders...")
    train_loader, val_loader, test_loader = build_all_dataloaders(batch_size=64)
    print("DataLoaders built successfully.\n")
    
    # Load pre-trained feature extractor and tokenizer
    print("Loading feature extractor and tokenizer...")
    vit_feature_extractor = ViTFeatureExtractor.from_pretrained("google/vit-base-patch16-224")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = tokenizer.vocab_size
    print("Feature extractor and tokenizer loaded.\n")
    
    # Hyperparameters 
    d_model = 768
    num_encoder_layers = 6
    num_decoder_layers = 6
    nhead = 8
    max_question_length = 32
    max_answer_length = 32
    num_epochs = 25
    learning_rate = 2e-5

    # Instantiate the model and print parameter counts.
    print("Initializing the model...")
    model = MedVQAModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        nhead=nhead,
        max_question_length=max_question_length,
        max_answer_length=max_answer_length
    ).to(device)
    total_params, trainable_params = count_parameters(model)
    print(f"Model initialized. Total parameters: {total_params:,}, Trainable: {trainable_params:,}\n")
    
    # Define optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    
    
    print("Sample predictions from validation set:")
    sample_and_log(model, val_loader, device, num_samples=5)

    
    print("Starting training...\n")
    for epoch in range(1, num_epochs + 1):
        print(f"=== Epoch {epoch}/{num_epochs} ===")
        
        
        (overall_train_loss, overall_train_acc,
         closed_train_loss, closed_train_acc,
         open_train_loss, open_train_acc,
         open_train_bleu, open_train_rouge) = train_epoch(model, train_loader, optimizer, device)
        
        
        
        (overall_val_loss, overall_val_acc,
         closed_val_loss, closed_val_acc,
         open_val_loss, open_val_acc,
         open_val_bleu, open_val_rouge) = evaluate_epoch(model, val_loader, device)
        
       
       
        print("Sample predictions from validation set:")
        sample_and_log(model, val_loader, device, num_samples=5)
        print("\n")

    # Save the trained model 
    model_save_path = "final_med_vqa_rad_model.pt"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model saved after training at '{model_save_path}'.\n")
    
    # Final evaluation on both Validation and Test sets
    print("Final Evaluation on Validation Set:")
    final_val_metrics = evaluate_epoch(model, val_loader, device)
    
    print("\nFinal Evaluation on Test Set:")
    final_test_metrics = evaluate_epoch(model, test_loader, device)
    
if __name__ == "__main__":
    main()