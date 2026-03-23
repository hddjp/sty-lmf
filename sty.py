from llamafactory.cli import main 
import sys

custom_args=["train","/mnt/shared-storage-gpfs2/ai4scifm-gpfs02/wuyixin/code/LLaMA-Factory/ds1.yaml"]

if __name__ == "__main__":
    sys.argv.extend(custom_args)
    main()