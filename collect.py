import pandas as pd
import numpy as np
import os
import json
from pathlib import Path
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="/workspace/yjy/results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)

    for mode in ["dev", "board"]:
        if not (output_dir / mode).exists():
            continue

        if mode == "board":
            continue
        filepaths = sorted((output_dir / mode).glob("*.json"), key=os.path.getmtime)

        frame = []
        for filepath in filepaths:
            data = json.load(open(filepath))
            curr = {}
            assert len(data["results"]) == 1
            task, result = list(data["results"].items())[0]
            curr["challenge"] = task.replace("challenge_", "")

            if "acc_norm" in result:
                curr["score"] = result["acc_norm"]
                curr["task"] = "single_choice"
            elif "mc2" in result:
                curr["score"] = result["mc2"]
                curr["task"] = "multiple_choice"
            elif "rougeL" in result:
                curr["score"] = result["rougeL"]
                curr["task"] = "summarization"
            frame.append(curr)

        frame.append(
            {
                "challenge": "average",
                "score": np.mean([f["score"] for f in frame]),
                "task": "-",
            }
        )

        frame = pd.DataFrame(frame)
        print(frame)
