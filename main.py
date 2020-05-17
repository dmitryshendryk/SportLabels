import argparse
import os 


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
       description='Text Reader')
    
    parser.add_argument('command',
                    metavar='<command>',
                    help="'train, detect, evaluate'")
   

    parser.add_argument('--device')

    args = parser.parse_args()

    if args.command == 'train':
        pass

    if args.command == 'evaluate':
        pass

    if args.command == 'detect':
        pass
    