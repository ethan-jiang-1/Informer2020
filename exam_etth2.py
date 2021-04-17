
#python -u main_informer.py --model informer --data ETTh2 --attn prob --freq h

from main_informer import main_loop, parse_args
import sys

sys.argv += "--model informer --data ETTh2 --attn prob --freq h".split(" ")
args = parse_args()

print(args)
main_loop(args)
