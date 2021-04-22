
#python -u main_informer.py --model informer --data ETTm1 --attn prob --freq t

from main_informer import main_loop, parse_args
import sys

sys.argv += "--model informer --data ETTm1 --attn prob --freq t".split(" ")
args = parse_args()

print(args)
main_loop(args)
