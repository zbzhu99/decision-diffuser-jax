import argparse
import importlib
import os
import sys

import absl

from utilities.utils import define_flags_with_default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("-g", type=int, default=0)
    args, unknown_flags = parser.parse_known_args()
    if args.g < 0:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.g)

    from utilities.utils import import_file

    config = getattr(import_file(args.config, "default_config"), "get_config")()
    config = define_flags_with_default(**config)
    absl.flags.FLAGS(sys.argv[:1] + unknown_flags)

    trainer = getattr(
        importlib.import_module("diffuser.trainer"), absl.flags.FLAGS.trainer
    )(config)
    trainer.train()


if __name__ == "__main__":
    main()
