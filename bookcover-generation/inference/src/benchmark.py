import argparse
import time
from contextlib import contextmanager
from types import SimpleNamespace

import torch
from transformers import AutoTokenizer

GENERATION_TEST_CASES = [
    ("제목", "저자", "출판사"),
    ("새로운 제목", "새로운 저자", "새로운 출판사"),
    ("테스트 제목", "테스트 저자", "테스트 출판사"),
    ("벤치마크 제목", "벤치마크 저자", "벤치마크 출판사"),
]


@contextmanager
def timer():
    record = SimpleNamespace(start_time=time.time())
    try:
        yield record
    finally:
        record.last_time = time.time()
        record.duration = record.last_time - record.start_time


@torch.no_grad()
def main(args: argparse.Namespace):
    print("[*] Load PyTorch and JIT models...")
    torch_model = torch.load(args.torch_model)
    jit_model = torch.jit.load(args.jit_model)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    for i, test_case in enumerate(GENERATION_TEST_CASES):
        encodings = f" {tokenizer.sep_token} ".join(test_case)
        encodings = tokenizer(encodings, return_tensors="pt")
        encodings = encodings.to("cuda" if args.use_gpu else "cpu")
        inputs = (encodings.input_ids, encodings.attention_mask)

        if i == 0:
            print("[*] Initialize the models")
            for _ in range(10):
                jit_model(*inputs, num_return_sequences=4)
                torch.cuda.synchronize()

        print(f"[*] Start benchmark of {i + 1}th case")

        for batch in [1, 4]:
            for repeats in [1, 10]:
                with timer() as record:
                    for _ in range(repeats):
                        torch_model(*inputs, num_return_sequences=batch)
                        torch.cuda.synchronize()
                print(
                    f"\t[*] PyTorch with batch={batch}, repeats={repeats}:"
                    f" avg {record.duration / repeats}"
                )

        for batch in [1, 4]:
            for repeats in [1, 10]:
                with timer() as record:
                    for _ in range(repeats):
                        jit_model(*inputs, num_return_sequences=batch)
                        torch.cuda.synchronize()
                print(
                    f"\t[*] TorchScript with batch={batch}, repeats={repeats}:"
                    f" {record.duration / repeats}"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-model", default="dalle-torch.pth")
    parser.add_argument("--jit-model", default="dalle-jit.pth")
    parser.add_argument("--tokenizer", default="dalle-16l-1024d")
    parser.add_argument("--use-gpu", action="store_true", default=False)
    main(parser.parse_args())
