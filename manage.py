# /// script
# requires-python = ">=3.9"
# dependencies = ["numpy", "torch", "Pillow", "matplotlib", "scikit-learn", "torchvision", "PyYAML"]
# ///
import argparse
from runManager import RunManager

def main():
    #TODO - use additional/better list methods, clean up, and merge args.delete and delete_folder. 
    parser = argparse.ArgumentParser(description="Run Manager CLI")
    parser.add_argument("--delete", type=str, help="Delete a specific sub-run")
    parser.add_argument("--list", nargs = "?", const = True, help="List existing run folders")
    parser.add_argument("--delete-folder", type = str, help = "Delete an entire experiment folder")
    parser.add_argument("--runs-dir", type=str, default="runs", help="Parent runs directory")

    args = parser.parse_args()
    rm = RunManager.utility_mode(base_dir=args.runs_dir)

    if args.list:
        print("[RunManager] Available runs:")
        rm.list_runsL()
        return

    if args.delete:
        run_path = rm.base_dir / args.delete
        confirm = input(f"Are you sure you want to delete '{run_path}'? (y/n): ")
        if confirm.lower().startswith("y"):
            rm.delete_run(args.delete)
    elif args.delete_folder:
        folder_path = rm.base_dir/ args.delete_folder
        confirm = input(f"Are you sure you want to delete the entire experiment folder '{folder_path}'? (y/n): ")
        if confirm.lower().startswith("y"):
            rm.delete_run(args.delete_folder)

if __name__ == "__main__":
    main()
