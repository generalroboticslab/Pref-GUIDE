import argparse

def get_config():
    parser = argparse.ArgumentParser(description="Training Configuration")

    # Add arguments
    parser.add_argument("--subject_id", type=int, default=37, help="Subject ID for data loading")
    parser.add_argument("--num_epochs", type=int, default=200, help="Number of epochs for training")
    parser.add_argument("--model_type", type=str, default="regression", help="Type of reward function")
    parser.add_argument("--game", type=str, default="hide_and_seek_1v1", help="which env to train")
    parser.add_argument("--pretrain", type=int, default=0, help="whether to use RL pretrain model")
    parser.add_argument("--freeze_encoder", type=int, default=0, help="whether to freeze encoder")
    parser.add_argument("--ref_model", type=str, default="regression", help="Type of reward model for reference")
    parser.add_argument("--batch_size", type=int, default=128, help="Batch size for training")
    parser.add_argument("--use_activation", type=int, default=0, help="whether to use RL pretrain model")
    
    parser.add_argument("--no_preference_window", type=int, default=1, help="whether to use no preference window")
    parser.add_argument("--moving_window", type=int, default=1, help="whether to sample pairs from moving window")
    
    return parser.parse_args()