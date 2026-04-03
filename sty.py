from llamafactory.cli import main 
import sys

custom_args=["train","/data/sty/sty-lmf/ds1.yaml"]

if __name__ == "__main__":
    sys.argv.extend(custom_args)
    main()