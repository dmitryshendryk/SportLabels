import argparse
import os 
import sys



ROOT = os.path.abspath('./')
sys.path.append(ROOT)

from imagenet.imagenet_train import train
from imagenet.visualize import visualize_model

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
       description='Text Reader')
    
    parser.add_argument('command',
                    metavar='<command>',
                    help="'train, detect, evaluate'")
   

    parser.add_argument('--device')
    parser.add_argument('--network')
    parser.add_argument('--weights')
    parser.add_argument('--save_weights')
    parser.add_argument('--folder_path')


    args = parser.parse_args()

    if args.command == 'train':
        if args.network == 'imagenet':
            train(args.device, args.save_weights)

    if args.command == 'process':
        pass

    if args.command == 'visualize':
        if args.network == 'imagenet':
            visualize_model(args.device, args.weights, 6)
    