"""B3: re-train WOS-R banks with ramp continuation (beta annealed 1->0 over 70 % of steps)."""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import b3_image as b
if __name__ == "__main__":
    from multiprocessing import Pool
    outp = "b3_results.json"
    jobs = [("imp", "WOS-R banco C=4 (5×5)", 0), ("imp", "WOS-R banco C=4 (3×3)", 0), ("gauss", "WOS-R banco C=4 (5×5)", 0), ("gauss", "WOS-R banco C=4 (3×3)", 0)]
    with Pool(2) as p:
        for r in p.imap_unordered(b.job, jobs):
            R = json.load(open(outp)); R.append(r); json.dump(R, open(outp, "w"), indent=0)
            print(r["regime"], r["model"], {k: round(v["psnr"], 2) for k, v in r["metrics"].items() if k.startswith("Set12")}, flush=True)
