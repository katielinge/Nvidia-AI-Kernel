"""
Puts the patched reference into the dataset definition (the copy the benchmark
uses) and into the seed kernel. Backs up both originals first.

Run from the Nvidia-AI-Kernel folder:
    python apply_patch.py C:\\Users\\Katie\\mlsys26-contest\\definitions\\moe\\moe_fp8_block_scale_ds_routing_topk8_ng8_kg4_e32_h7168_i2048.json
To undo:
    python apply_patch.py --restore <same path>
"""
import json
import pathlib
import shutil
import sys

HERE = pathlib.Path(__file__).parent
BACKUP_DIR = HERE / "original_reference_backup"   # kept OUTSIDE the dataset folder
SEED = HERE / "solution" / "triton" / "kernel.py"


def main():
    args = sys.argv[1:]
    restore = "--restore" in args
    args = [a for a in args if a != "--restore"]
    if len(args) != 1:
        sys.exit(__doc__)
    defn = pathlib.Path(args[0])
    backup_json = BACKUP_DIR / defn.name
    backup_seed = BACKUP_DIR / "kernel.py"

    if restore:
        shutil.copy2(backup_json, defn)
        if backup_seed.exists():
            shutil.copy2(backup_seed, SEED)
        print("Restored the original reference and seed kernel.")
        return

    data = json.loads(defn.read_text(encoding="utf-8"))
    if "reference" not in data or "def run(" not in data["reference"]:
        sys.exit("Could not find the reference code in the 'reference' field. Send this to Claude.")

    BACKUP_DIR.mkdir(exist_ok=True)
    if not backup_json.exists():          # never overwrite the true original
        shutil.copy2(defn, backup_json)
        if SEED.exists():
            shutil.copy2(SEED, backup_seed)

    patched = (HERE / "patched_reference.py").read_text(encoding="utf-8")
    data["reference"] = patched
    defn.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    # Seed kernel for the agent = patched reference with run -> kernel
    SEED.write_text(patched.replace("def run(", "def kernel(", 1), encoding="utf-8")

    print("Patched reference written to:", defn)
    print("Seed kernel updated:", SEED)
    print("Originals backed up in:", BACKUP_DIR)


if __name__ == "__main__":
    main()
