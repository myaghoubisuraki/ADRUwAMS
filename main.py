import argparse
from train_utils import main
import warnings
warnings.simplefilter("ignore", UserWarning)

def parse_args():
    parser = argparse.ArgumentParser(description='Train the 3D Medical Imaging Model')
    
    # Model selection
    parser.add_argument('--model', type=str, choices=['ADRUwAMS'], required=True, help='Choose the model you want to use.')
    
    # Batch size
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training.')
    
    # Number of epochs
    parser.add_argument('--epochs', type=int, default=1, help='Number of epochs for training.')
    
    # Pretrained model
    parser.add_argument('--pretrained', type=str, default=None, help='Path to a pretrained model file under the pre_weights directory.')
    
    # Learning Rate
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate for the optimizer.')

    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(model_name=args.model, batch_size=args.batch_size, lr=args.lr, num_epochs=args.epochs, pretrained=args.pretrained)

