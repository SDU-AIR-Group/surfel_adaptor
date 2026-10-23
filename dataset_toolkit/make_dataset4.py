# Legacy monolithic pipeline retained for reference only.










































import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import fcntl
import socket
import tempfile


try:
    import trimesh  
    HAS_TRIMESH = True
except Exception:
    HAS_TRIMESH = False


INDEX_HEADER = [
    "terrain_source", "terrain_tag", "eta",
    "patch_file", "patch_id",
    "scan_id", "scan_path",
    "voxel_id", "voxel_path",
    "feat_id", "feat_path",
    "render_subdir_id", "render_dir",
    "full_feat_id", "full_feat_path",
    "timestamp",
]
LOG_TAIL_LINES_DEFAULT = 3


def now_str() -> str:
    return dt.datetime.now().strftime("%Y%m%d-%H%M%S")

def start_tag_str() -> str:
    return dt.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

def short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def seconds_to_hms(secs: float) -> str:
    secs = int(secs)
    return f"{secs//3600:02d}:{(secs%3600)//60:02d}:{secs%60:02d}"

def tail_log_print(log_file: Path, n: int = LOG_TAIL_LINES_DEFAULT):
    if not log_file.exists():
        print("  <no log file>"); return
    try:
        lines = log_file.read_text(encoding="utf-8", errors="ignore").splitlines()
        for ln in lines[-n:]:
            print("  " + ln)
    except Exception as e:
        print(f"  <failed to read log: {e}>")

def _unique_tmp_for(target_path: Path) -> Path:

    d = target_path.parent
    base = target_path.stem
    
    return d / f"{base}.tmp.{os.getpid()}.{time.time_ns()}"

def run_cmd(cmd: List[str], log_file: Path, timeout: int = 0, idle_timeout: int = 0, verbose: bool = False) -> int:

    ensure_dir(log_file.parent)
    with log_file.open("w", encoding="utf-8") as lf:
        lf.write("$ " + " ".join(cmd) + "\n\n"); lf.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE if verbose else lf,
            stderr=subprocess.STDOUT,
            text=True if verbose else False,
            bufsize=1,
            start_new_session=True
        )
        start = time.time()
        last_size = 0
        last_growth = time.time()

        def kill(sig):
            try: os.killpg(proc.pid, sig)
            except Exception: pass

        rc = None
        try:
            while True:
                if verbose and proc.stdout:
                    line = proc.stdout.readline()
                    if line:
                        sys.stdout.write(line)
                        lf.write(line); lf.flush()
                        last_growth = time.time()

                rc = proc.poll()
                if rc is not None: break

                time.sleep(2)

                if timeout and time.time()-start > timeout:
                    lf.write(f"\n[watchdog] TIMEOUT>{timeout}s, SIGTERM...\n"); lf.flush()
                    kill(signal.SIGTERM); time.sleep(2)
                    if proc.poll() is None:
                        lf.write("[watchdog] still alive, SIGKILL.\n"); lf.flush()
                        kill(signal.SIGKILL)
                    rc = 124; break

                if idle_timeout:
                    try: sz = log_file.stat().st_size
                    except Exception: sz = 0
                    if sz != last_size:
                        last_size = sz; last_growth = time.time()
                    elif time.time()-last_growth > idle_timeout:
                        lf.write(f"\n[watchdog] IDLE>{idle_timeout}s, SIGTERM...\n"); lf.flush()
                        kill(signal.SIGTERM); time.sleep(2)
                        if proc.poll() is None:
                            lf.write("[watchdog] still alive, SIGKILL.\n"); lf.flush()
                            kill(signal.SIGKILL)
                        rc = 125; break

            if verbose and proc.stdout:
                for line in proc.stdout:
                    sys.stdout.write(line); lf.write(line)
            lf.flush()
        finally:
            if proc.poll() is None:
                kill(signal.SIGKILL)

    return rc if rc is not None else proc.returncode

def link_or_copy(src: Path, dst: Path):
    ensure_dir(dst.parent)
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)

def rel_to_3voxel(root: Path, p: Optional[Path]) -> str:
    if p is None:
        return ""
    base = root / "3_voxel"
    try:
        return str(p.relative_to(base))
    except Exception:
        return str(p)
    

def index_dirs(root: Path) -> Tuple[Path, Path, Path]:

    idx_dir = root / "3_voxel" / "index"
    master = idx_dir / "index.csv"
    shards = idx_dir / "shards"
    return idx_dir, master, shards


def read_csv_rows_locked(csv_path: Path) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    if not csv_path.exists():
        return rows
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_SH)
        try:
            
            f.seek(0, os.SEEK_END)
            if f.tell() == 0:
                return []
            f.seek(0)
            r = csv.DictReader(f)
            header = r.fieldnames or []
            if header != INDEX_HEADER:
                raise SystemExit(
                    f"[ERROR] index.csv header mismatch!\n"
                    f"  expected: {INDEX_HEADER}\n"
                    f"  actual:   {header}\n"
                    "Check the CSV header or use the matching script version."
                )
            rows = list(r)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    return rows


def atomic_write_master(csv_path: Path, rows: List[Dict[str, str]]):
    
    tmp = _unique_tmp_for(csv_path)
    ensure_dir(csv_path.parent)
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_HEADER)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in INDEX_HEADER})
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, csv_path)  


def ensure_master_initialized(master_csv: Path):





    ensure_dir(master_csv.parent)
    
    with master_csv.open("a+", newline="", encoding="utf-8") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        try:
            f.seek(0, os.SEEK_END)
            sz = f.tell()
            if sz == 0:
                
                tmp = master_csv.with_suffix(".tmp")
                with tmp.open("w", newline="", encoding="utf-8") as tf:
                    w = csv.DictWriter(tf, fieldnames=INDEX_HEADER)
                    w.writeheader()
                    tf.flush(); os.fsync(tf.fileno())
                os.replace(tmp, master_csv)
                return

            
            f.seek(0)
            r = csv.DictReader(f)
            header = r.fieldnames or []
            if header != INDEX_HEADER:
                raise SystemExit(
                    f"[ERROR] index.csv header mismatch!\n"
                    f"  expected: {INDEX_HEADER}\n"
                    f"  actual:   {header}\n"
                    "Check the CSV header or use the matching script version."
                )
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)


def append_csv_row_shard(shard_csv: Path, row: Dict[str, str]):

    ensure_dir(shard_csv.parent)
    write_header = (not shard_csv.exists()) or (shard_csv.stat().st_size == 0)
    with shard_csv.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=INDEX_HEADER)
        if write_header:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in INDEX_HEADER})

def merge_shard_into_master(master_csv: Path, shard_csv: Path):

    if not shard_csv.exists() or shard_csv.stat().st_size == 0:
        return  

    archive_dir = shard_csv.parent / "archived"
    ensure_dir(archive_dir)

    
    with shard_csv.open("r", newline="", encoding="utf-8") as sf:
        sr = csv.DictReader(sf)
        s_header = sr.fieldnames or []
        if s_header != INDEX_HEADER:
            raise SystemExit(f"[ERROR] shard header mismatch: {shard_csv}\n expected {INDEX_HEADER}, got {s_header}")
        shard_rows = list(sr)

    
    ensure_dir(master_csv.parent)
    with master_csv.open("a+", newline="", encoding="utf-8") as mf:
        fcntl.flock(mf.fileno(), fcntl.LOCK_EX)
        try:
            mf.seek(0, os.SEEK_END)
            empty = (mf.tell() == 0)
            if empty:
                
                atomic_write_master(master_csv, [])
            
            master_rows = []
            if master_csv.exists() and master_csv.stat().st_size > 0:
                with master_csv.open("r", newline="", encoding="utf-8") as rf:
                    rr = csv.DictReader(rf)
                    r_header = rr.fieldnames or []
                    if r_header != INDEX_HEADER:
                        raise SystemExit(f"[ERROR] master header mismatch: {master_csv}")
                    master_rows = list(rr)
            existing = {r["scan_id"] for r in master_rows if r.get("scan_id")}
            added = 0
            for r in shard_rows:
                sid = r.get("scan_id", "")
                if sid and sid not in existing:
                    master_rows.append(r)
                    existing.add(sid)
                    added += 1
            atomic_write_master(master_csv, master_rows)
            print(f"[merge] {shard_csv.name}: +{added} rows (master now {len(master_rows)})")
        finally:
            fcntl.flock(mf.fileno(), fcntl.LOCK_UN)

    
    dest = archive_dir / shard_csv.name
    if dest.exists():
        dest.unlink()
    shutil.move(str(shard_csv), str(dest))

def merge_unarchived_shards(root: Path):

    idx_dir, master_csv, shards_dir = index_dirs(root)
    ensure_dir(shards_dir)
    to_merge = [p for p in sorted(shards_dir.glob("index_shard_*.csv")) if p.is_file()]
    if not to_merge:
        return
    print(f"[merge] found {len(to_merge)} shard(s) to merge at startup")
    for shard in to_merge:
        try:
            merge_shard_into_master(master_csv, shard)
        except Exception as e:
            print(f"[merge] ERROR merging {shard.name}: {e}")


def _locks_dir(root: Path, stage: str) -> Path:

    d = root / "3_voxel" / "index" / "locks" / stage
    d.mkdir(parents=True, exist_ok=True)
    return d

def _lock_path_for(root: Path, stage: str, scan_id: str) -> Path:

    suffix = scan_id.split("_")[-1] if "_" in scan_id else short_hash(scan_id)
    return _locks_dir(root, stage) / f"scan_{suffix}.lock"

def _process_alive(pid: int) -> bool:

    try:
        if pid <= 0: 
            return False
        os.kill(pid, 0)       
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True           
    except Exception:
        return True

def _lock_is_stale(lock_meta: dict, ttl_hours: float) -> bool:

    ts = float(lock_meta.get("ts", 0.0) or 0.0)
    return (time.time() - ts) > (ttl_hours * 3600.0)

def _acquire_row_lock(lock_path: Path, ttl_hours: float = 6.0) -> bool:




    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        
        try:
            with lock_path.open("r", encoding="utf-8") as f:
                meta = json.load(f)
            pid = int(meta.get("pid", -1))
            if (not _process_alive(pid)) or _lock_is_stale(meta, ttl_hours):
                
                try: lock_path.unlink()
                except FileNotFoundError: pass
                try:
                    fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                except FileExistsError:
                    return False
            else:
                return False
        except Exception:
            
            return False
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "host": socket.gethostname(), "ts": time.time()}, f)
        f.flush(); os.fsync(f.fileno())
    return True

def _release_row_lock(lock_path: Path):

    try: lock_path.unlink()
    except FileNotFoundError: pass

def _update_master_row(master_csv: Path, scan_id: str, updates: Dict[str, str]):








    ensure_dir(master_csv.parent)
    with master_csv.open("a+", newline="", encoding="utf-8") as f_ex:
        fcntl.flock(f_ex.fileno(), fcntl.LOCK_EX)
        try:
            
            f_ex.seek(0, os.SEEK_END)
            if f_ex.tell() == 0:
                tmp0 = _unique_tmp_for(master_csv)
                with tmp0.open("w", newline="", encoding="utf-8") as tf:
                    w = csv.DictWriter(tf, fieldnames=INDEX_HEADER)
                    w.writeheader()
                    tf.flush(); os.fsync(tf.fileno())
                os.replace(tmp0, master_csv)

            
            with master_csv.open("r", newline="", encoding="utf-8") as rf:
                rr = csv.DictReader(rf)
                header = rr.fieldnames or []
                if header != INDEX_HEADER:
                    raise SystemExit(f"[ERROR] master header mismatch: {header}")
                rows = list(rr)

            
            found = False
            for r in rows:
                if r.get("scan_id") == scan_id:
                    r.update({k: str(v) for k, v in updates.items()})
                    found = True
                    break
            if not found:
                raise SystemExit(f"[ERROR] scan_id not found in master: {scan_id}")

            
            tmp = _unique_tmp_for(master_csv)
            with tmp.open("w", newline="", encoding="utf-8") as tf:
                w = csv.DictWriter(tf, fieldnames=INDEX_HEADER)
                w.writeheader()
                for row in rows:
                    w.writerow({k: row.get(k, "") for k in INDEX_HEADER})
                tf.flush(); os.fsync(tf.fileno())
            os.replace(tmp, master_csv)
        finally:
            fcntl.flock(f_ex.fileno(), fcntl.LOCK_UN)



def _is_oom_error(log_file: Path, rc: int) -> bool:





    try:
        txt = log_file.read_text(encoding="utf-8", errors="ignore")[-20000:]  
        low = txt.lower()
    except Exception:
        low = ""
    patterns = [
        "cuda out of memory",
        "runtimeerror: cuda out of memory",
        "out of memory",
        "std::bad_alloc",
        "cublas_status_alloc_failed",
        "cudnn_status_alloc_failed",
        "cumemalloc",
        "resource exhausted",
        "killed",
    ]
    if any(p in low for p in patterns):
        return True
    return rc in (9, 137)



def step1_aux_preproc(args):
    root = Path(args.root)
    toolkit = Path(args.toolkit)
    blender = args.blender
    input_root = root / args.aux_raw_dir
    out_dir = root / "0_aux_models_preproc"
    log_dir = root / "3_voxel" / "logs" / "step1"
    py = toolkit / "preprocess" / "0aux_preproc.py"

    ensure_dir(out_dir); ensure_dir(log_dir)
    start_time = time.time()
    total = args.end_aux - args.begin_aux + 1
    done_cnt = 0

    for i in range(args.begin_aux, args.end_aux + 1):
        name = f"000-{i:03d}"
        in_dir = input_root / name
        if not in_dir.is_dir():
            print(f"[step1] skip {name}: not found")
            continue
        done_flag = out_dir / "_logs" / f"{name}.done"
        if done_flag.exists():
            print(f"[step1] done {name}, skip")
            done_cnt += 1
            continue
        log = log_dir / f"{name}.log"
        cmd = [blender, "-b", "-P", str(py), "--",
               "--input_folder", str(in_dir),
               "--output_folder", str(out_dir),
               "--save_mesh"]
        t0 = time.time()
        rc = run_cmd(cmd, log, timeout=args.step_timeout, idle_timeout=args.idle_timeout, verbose=args.verbose)
        t1 = time.time()
        if rc == 0:
            ensure_dir(done_flag.parent); done_flag.touch()
            done_cnt += 1
            print(f"[#] step1 {done_cnt}/{total} | {name} | t_step={t1-t0:.1f}s | t_total={seconds_to_hms(time.time()-start_time)}")
        else:
            print(f"[step1] ERROR {name} (rc={rc}) (see {log})")
            tail_log_print(log, args.log_tail_lines)

    print(f"[step1] DONE. finished={done_cnt}/{total}  elapsed={seconds_to_hms(time.time()-start_time)}")


def step2_terrain_preproc(args):
    root = Path(args.root)
    toolkit = Path(args.toolkit)
    blender = args.blender
    out_root = root / "0_terrain_models_preproc"
    log_dir = root / "3_voxel" / "logs" / "step2"
    py = toolkit / "preprocess" / "0terrain_preproc.py"

    ensure_dir(out_root); ensure_dir(log_dir)

    terrain_dirs = [x.strip() for x in (args.terrain_dir or "").split(",") if x.strip()]
    if not terrain_dirs:
        raise SystemExit("[ERROR] --terrain_dir is empty. Example: 'terrain_models/offroad,terrain_models/city'")

    start_time = time.time()
    total = 0; done_cnt = 0

    
    for td in terrain_dirs:
        src = root / td
        total += len(list(src.glob("*.glb")))

    for td in terrain_dirs:
        src = root / td
        if not src.exists():
            print(f"[step2] WARN terrain dir {src} not found, skip")
            continue
        prefix = Path(td).name  
        for glb in sorted(src.glob("*.glb")):
            terrain_tag = f"{prefix}_{glb.stem}"
            out_dir = out_root / terrain_tag
            out_model = out_dir / "model.glb"
            done_flag = out_dir / ".done"
            if out_model.exists() and done_flag.exists():
                done_cnt += 1
                print(f"[step2] done {terrain_tag}, skip")
                continue
            ensure_dir(out_dir)
            log = log_dir / f"{terrain_tag}.log"
            cmd = [blender, "-b", "-P", str(py), "--",
                   "--object", str(glb),
                   "--output_folder", str(out_dir),
                   "--save_mesh"]
            t0 = time.time()
            rc = run_cmd(cmd, log, timeout=args.step_timeout, idle_timeout=args.idle_timeout, verbose=args.verbose)
            t1 = time.time()
            if rc == 0 and out_model.exists():
                ensure_dir(done_flag.parent); done_flag.touch()
                done_cnt += 1
                print(f"[#] step2 {done_cnt}/{total} | {terrain_tag} | t_step={t1-t0:.1f}s | t_total={seconds_to_hms(time.time()-start_time)}")
            else:
                print(f"[step2] ERROR {terrain_tag} (rc={rc}) (see {log})")
                tail_log_print(log, args.log_tail_lines)

    print(f"[step2] DONE. finished={done_cnt}/{total}  elapsed={seconds_to_hms(time.time()-start_time)}")


def _load_dims(glb: Path) -> Optional[Tuple[float,float,float]]:
    if not HAS_TRIMESH:
        return None
    try:
        m = trimesh.load(str(glb), force='mesh')
        mn, mx = m.bounds
        return float(mx[0]-mn[0]), float(mx[1]-mn[1]), float(mx[2]-mn[2])
    except Exception:
        return None

def step3_terrain_split(args):
    if not HAS_TRIMESH:
        print("[step3] WARN: trimesh is unavailable; size filtering is skipped (pip install trimesh)")

    root = Path(args.root)
    toolkit = Path(args.toolkit)
    blender = args.blender
    in_root = root / "0_terrain_models_preproc"
    out_root = root / "1_terrain_splitting"
    log_dir = root / "3_voxel" / "logs" / "step3"
    py = toolkit / "preprocess" / "1terrain_splitting.py"

    ensure_dir(out_root); ensure_dir(log_dir)
    terrains = sorted([d for d in in_root.iterdir() if (d/"model.glb").exists()])

    start_time = time.time()
    total = len(terrains); done_cnt = 0
    print("&&&&&&&&&&&")
    print(f"patch_filter_condition:  aspect<= {args.patch_aspect_max}, min_edge>= {args.patch_min_edge} m")
    print("&&&&&&&&&&&")

    for d in terrains:
        terrain_tag = d.name
        obj = d / "model.glb"
        out_dir = out_root / terrain_tag
        patches_dir = out_dir / "patches"
        done_flag = out_dir / ".done"

        if patches_dir.exists() and done_flag.exists() and list(patches_dir.glob("patch_*.glb")):
            done_cnt += 1
            print(f"[step3] done {terrain_tag}, skip split")
        else:
            ensure_dir(out_dir)
            log = log_dir / f"{terrain_tag}.log"
            cmd = [blender, "-b", "-P", str(py), "--",
                   "--object", str(obj),
                   "--output_folder", str(out_dir),
                   "--lidar_height", str(args.lidar_height),
                   "--save_mesh"]
            t0 = time.time()
            rc = run_cmd(cmd, log, timeout=args.step_timeout, idle_timeout=args.idle_timeout, verbose=args.verbose)
            t1 = time.time()
            if rc != 0:
                print(f"[step3] ERROR {terrain_tag} (rc={rc}) (see {log})")
                tail_log_print(log, args.log_tail_lines)
                continue
            ensure_dir(done_flag.parent); done_flag.touch()
            done_cnt += 1
            print(f"[#] step3 split {done_cnt}/{total} | {terrain_tag} | t_step={t1-t0:.1f}s | t_total={seconds_to_hms(time.time()-start_time)}")

        
        patches = sorted(patches_dir.glob("patch_*.glb"),
                         key=lambda p: int(re.findall(r"patch_(\d+)", p.name)[0]) if re.findall(r"patch_(\d+)", p.name) else 0)
        if not patches:
            print(f"[step3] WARN no patches for {terrain_tag}")
            continue

        
        per_csv = out_dir / "patch_filter.csv"
        global_csv = out_root / "patch_filter_index.csv"
        per_header = ["terrain_tag","patch_file","dx","dy","dz","aspect","min_edge","status","reason"]
        ensure_dir(per_csv.parent)
        new_per = not per_csv.exists()
        new_glo = not global_csv.exists()

        kept = []
        with per_csv.open("a", newline="", encoding="utf-8") as pf, \
             global_csv.open("a", newline="", encoding="utf-8") as gf:
            pw = csv.DictWriter(pf, fieldnames=per_header)
            gw = csv.DictWriter(gf, fieldnames=per_header)
            if new_per: pw.writeheader()
            if new_glo: gw.writeheader()



            for p in patches:
                if HAS_TRIMESH:
                    dims = _load_dims(p)
                else:
                    dims = None
                if dims is None:
                    row = {"terrain_tag":terrain_tag,"patch_file":str(p),
                           "dx":"","dy":"","dz":"","aspect":"","min_edge":"",
                           "status":"kept","reason":"no_dims"}
                    pw.writerow(row); gw.writerow(row)
                    kept.append(p); continue

                dx,dz,dy = dims
                aspect = max(dx,dy)/max(1e-9, min(dx,dy))
                min_edge = min(dx,dy)
                ok = (aspect <= args.patch_aspect_max) and (min_edge >= args.patch_min_edge)
                status = "kept" if ok else "rejected"
                reason = "" if ok else (f"aspect>{args.patch_aspect_max}" if aspect>args.patch_aspect_max else f"min_edge<{args.patch_min_edge}")
                row = {"terrain_tag":terrain_tag,"patch_file":str(p),
                       "dx":f"{dx:.3f}","dy":f"{dy:.3f}","dz":f"{dz:.3f}",
                       "aspect":f"{aspect:.3f}","min_edge":f"{min_edge:.3f}",
                       "status":status,"reason":reason}
                pw.writerow(row); gw.writerow(row)

                if ok or args.reject_action=="keep":
                    kept.append(p)
                else:
                    rej_dir = out_dir / "patches_rejected"
                    ensure_dir(rej_dir)
                    if args.reject_action == "move":
                        shutil.move(str(p), str(rej_dir / p.name))
                    elif args.reject_action == "delete":
                        p.unlink(missing_ok=True)

        kept_final = [pp for pp in sorted(patches_dir.glob("patch_*.glb")) if pp.exists()]
        print(f"-----------    [step3] filter {terrain_tag}: kept {len(kept_final)}/{len(patches)}   ------------")
        print("  ")
            

    print(f"######  [step3] DONE. processed={done_cnt}/{total}  elapsed={seconds_to_hms(time.time()-start_time)}    ######")


def enumerate_patches(root: Path) -> List[Tuple[str, int, Path]]:

    base = root / "1_terrain_splitting"
    items: List[Tuple[str, int, Path]] = []
    if not base.is_dir():
        print(f"[WARN] {base} not found. Complete Step3 first.")
        return items
    for tdir in sorted(base.iterdir()):
        patches_dir = tdir / "patches"
        if not patches_dir.is_dir(): continue
        for glb in sorted(patches_dir.glob("patch_*.glb"),
                          key=lambda p: int(re.findall(r"patch_(\d+)", p.name)[0]) if re.findall(r"patch_(\d+)", p.name) else 0):
            m = re.findall(r"patch_(\d+)", glb.name)
            idx = int(m[0]) if m else 0
            items.append((tdir.name, idx, glb))
    return items


def step4_scan(args):
    root = Path(args.root)
    toolkit = Path(args.toolkit)
    blender = args.blender
    target = args.target_samples
    if target <= 0:
        raise SystemExit("[ERROR] --mode step4 requires --target_samples > 0")

    
    idx_dir, master_csv, shards_dir = index_dirs(root)
    ensure_dir(idx_dir); ensure_dir(shards_dir); ensure_dir(shards_dir / "archived")
    merge_unarchived_shards(root)

    
    start_tag = start_tag_str()
    shard_csv = shards_dir / f"index_shard_{start_tag}_pid{os.getpid()}.csv"

    patches = enumerate_patches(root)
    if not patches:
        raise SystemExit("[ERROR] No patches found; complete Step3 and check 1_terrain_splitting/*/patches/.")

    success = 0
    start_time = time.time()
    i = 0
    total_patches = len(patches)
    print(f"[step4] patches={total_patches}  target_samples={target}")
    print(f"[step4] shard: {shard_csv}")

    while success < target:
        terrain_tag, pidx, patch_path = patches[i % total_patches]
        i += 1

        patch_id = f"{terrain_tag}_p{pidx}"
        suffix = short_hash(f"{patch_id}|{now_str()}|{os.getpid()}|{patch_path}")
        scan_id = f"scan_{terrain_tag}_{pidx}_{suffix}"

        
        work = root / "temp" / "step4" / suffix
        single = work / "terrain_singleton"
        scans_dir = work / "scans"
        ensure_dir(scans_dir); ensure_dir(single)
        try:
            (single / patch_path.name).symlink_to(patch_path.resolve())
        except Exception:
            shutil.copy2(patch_path, single / patch_path.name)

        
        log = root / "3_voxel" / "logs" / "step4" / f"{scan_id}.log"
        ensure_dir(log.parent)
        py = toolkit / "tools" / "aux_scanning.py"
        cmd = [blender, "-b", "-P", str(py), "--",
               "--aux_model_dir", str(root / "0_aux_models_preproc"),
               "--terrain_model_dir", str(single),
               "--output_folder", str(work),
               "--scale_factor", *map(str, args.scale_factor),
               "--random_pose_rate", str(args.random_pose_rate),
               "--lidar_height", str(args.lidar_height),
               "--save_mesh"]
        t0 = time.time()
        rc = run_cmd(cmd, log, timeout=args.step_timeout, idle_timeout=args.idle_timeout, verbose=args.verbose)
        t1 = time.time()

        scan_npy = scans_dir / "scan.npy"
        if rc != 0 or not scan_npy.exists():
            print()
            print(f"[step4] FAIL patch={patch_id}  rc={rc}  (see {log})")
            print()
            tail_log_print(log, args.log_tail_lines)
            shutil.rmtree(work, ignore_errors=True)
            continue

        
        meta_path = scans_dir / "scan_meta.json"
        scan_type, eta = "unknown", ""
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                scan_type = str(meta.get("scan_type", "unknown"))
                v = meta.get("eta")
                if v is not None:
                    try: eta = f"{float(v):.6f}"
                    except Exception: eta = str(v)
            except Exception:
                pass

        
        scans_out = root / "3_voxel" / "scans"
        ensure_dir(scans_out)
        scan_dst = scans_out / f"{scan_id}.npy"
        link_or_copy(scan_npy, scan_dst)

        
        terrain_source = terrain_tag.split("_", 1)[0] if "_" in terrain_tag else ""
        row = {
            "terrain_source": terrain_source,
            "terrain_tag": terrain_tag,
            "eta": eta,
            "patch_file": str(patch_path),
            "patch_id": patch_id,
            "scan_id": scan_id,
            "scan_path": rel_to_3voxel(root, scan_dst),
            "voxel_id": "",
            "voxel_path": "",
            "feat_id": "",
            "feat_path": "",
            "render_subdir_id": "",
            "render_dir": "",
            "full_feat_id": "",
            "full_feat_path": "",
            "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        }
        append_csv_row_shard(shard_csv, row)

        success += 1
        print(f"[#] step4 {success}/{target} | patch={patch_id} | scan={scan_id} | type={scan_type} | eta={eta or '-'} "
              f"| t_step={t1-t0:.1f}s | t_total={seconds_to_hms(time.time()-start_time)}")
        shutil.rmtree(work, ignore_errors=True)

    
    try:
        merge_shard_into_master((root / "3_voxel" / "index" / "index.csv"), shard_csv)
    except Exception as e:
        print(f"[merge] ERROR merging own shard: {e}")

    print(f"[step4] DONE. success={success}/{target}  elapsed={seconds_to_hms(time.time()-start_time)}")


def step5_voxel(args):









    root = Path(args.root)
    toolkit = Path(args.toolkit)
    master_csv = root / "3_voxel" / "index" / "index.csv"
    ensure_master_initialized(master_csv) 
    if not master_csv.exists():
        raise SystemExit("[ERROR] Missing index/index.csv; complete the Step4 merge first.")

    py = toolkit / "voxel.py"
    done = 0
    t0_all = time.time()

    while True:
        
        rows = read_csv_rows_locked(master_csv)
        candidates = [r for r in rows if r.get("scan_id") and not r.get("voxel_id")]
        if not candidates:
            if done == 0:
                print("[step5] nothing to do.")
            break

        
        picked = None
        for r in candidates:
            lk = _lock_path_for(root, "step5", r["scan_id"])
            if _acquire_row_lock(lk, ttl_hours=6.0):
                picked = (r, lk); break
        if not picked:
            time.sleep(0.2); continue

        r, lk = picked
        scan_id = r["scan_id"]
        suffix = scan_id.split("_")[-1] if "_" in scan_id else short_hash(scan_id)
        voxel_id = f"voxel_{suffix}"

        
        work = root / "temp" / "step5" / suffix
        scans_dir = work / "scans"
        ensure_dir(scans_dir)

        scan_src = root / "3_voxel" / r["scan_path"]
        if not scan_src.exists():
            print(f"[step5] WARN scan missing: {scan_src} (skip)")
            _release_row_lock(lk); shutil.rmtree(work, ignore_errors=True); continue
        link_or_copy(scan_src, scans_dir / "scan.npy")

        log = root / "3_voxel" / "logs" / "step5" / f"{voxel_id}.log"
        ensure_dir(log.parent)
        cmd = [sys.executable, str(py), "--input_dir", str(work), "--output_dir", str(work)]
        tt0 = time.time()
        rc = run_cmd(cmd, log, timeout=args.step_timeout, idle_timeout=args.idle_timeout, verbose=args.verbose)
        tt1 = time.time()

        voxel_src = work / "voxels" / "voxel.ply"
        if rc != 0 or not voxel_src.exists():
            print(f"[step5] FAIL scan={scan_id} rc={rc} (see {log})")
            tail_log_print(log, args.log_tail_lines)
            _release_row_lock(lk); shutil.rmtree(work, ignore_errors=True); continue

        
        voxels_out = root / "3_voxel" / "voxels"
        ensure_dir(voxels_out)
        voxel_dst = voxels_out / f"{voxel_id}.ply"
        link_or_copy(voxel_src, voxel_dst)

        
        _update_master_row(master_csv, scan_id, {
            "voxel_id": voxel_id,
            "voxel_path": rel_to_3voxel(root, voxel_dst),
        })

        
        _release_row_lock(lk)
        shutil.rmtree(work, ignore_errors=True)

        done += 1
        print(f"[#] step5 +1 | scan={scan_id} -> voxel={voxel_id} | t_step={tt1-tt0:.1f}s | t_total={seconds_to_hms(time.time()-t0_all)}")

    print(f"[step5] DONE. finished={done}")


def step6_feature(args):






    root = Path(args.root)
    toolkit = Path(args.toolkit)
    master_csv = root / "3_voxel" / "index" / "index.csv"
    ensure_master_initialized(master_csv) 
    if not master_csv.exists():
        raise SystemExit("[ERROR] Missing index/index.csv; complete Step5 first.")

    py = toolkit / "extract_feature.py"
    done = 0
    t0_all = time.time()

    max_oom = max(0, int(getattr(args, "step6_oom_retries", 3)))
    backoff = max(0, int(getattr(args, "step6_oom_backoff", 30)))
    local_cooldown_sec = max(0, int(getattr(args, "step6_local_cooldown", 300)))

    
    local_cooldown: Dict[str, float] = {}

    while True:
        
        rows = read_csv_rows_locked(master_csv)
        candidates = [r for r in rows if r.get("voxel_id") and not r.get("feat_id")]
        if not candidates:
            if done == 0:
                print("[step6] nothing to do.")
            break

        
        now = time.time()
        candidates = [r for r in candidates if local_cooldown.get(r["scan_id"], 0) <= now]
        if not candidates:
            time.sleep(0.5)
            continue

        
        picked = None
        for r in candidates:
            lk = _lock_path_for(root, "step6", r["scan_id"])
            if _acquire_row_lock(lk, ttl_hours=6.0):
                picked = (r, lk); break
        if not picked:
            time.sleep(0.2)
            continue

        r, lk = picked
        scan_id = r["scan_id"]
        suffix = scan_id.split("_")[-1] if "_" in scan_id else short_hash(scan_id)
        feat_id = f"feat_{suffix}"

        
        attempts = 0
        success = False
        while attempts <= max_oom and not success:
            attempts += 1

            
            work = root / "temp" / "step6" / f"{suffix}_try{attempts}"
            scans_dir = work / "scans"
            vox_dir = work / "voxels"
            ensure_dir(scans_dir); ensure_dir(vox_dir)

            scan_src  = root / "3_voxel" / r["scan_path"]
            voxel_src = root / "3_voxel" / r["voxel_path"]
            if not scan_src.exists() or not voxel_src.exists():
                print(f"[step6] WARN missing inputs. scan={scan_src.exists()} voxel={voxel_src.exists()} (skip in this proc)")
                shutil.rmtree(work, ignore_errors=True)
                break  

            link_or_copy(scan_src,  scans_dir / "scan.npy")
            link_or_copy(voxel_src, vox_dir   / "voxel.ply")

            log = root / "3_voxel" / "logs" / "step6" / f"{feat_id}.log"
            ensure_dir(log.parent)
            cmd = [sys.executable, str(py), "--input_dir", str(work), "--output_dir", str(work)]

            t0 = time.time()
            rc = run_cmd(cmd, log, timeout=args.step_timeout, idle_timeout=args.idle_timeout, verbose=args.verbose)
            t1 = time.time()

            feat_src = work / "features" / "sonata" / "feat.npz"
            if rc == 0 and feat_src.exists():
                
                feats_out = root / "3_voxel" / "features"
                ensure_dir(feats_out)
                feat_dst = feats_out / f"{feat_id}.npz"
                link_or_copy(feat_src, feat_dst)

                _update_master_row(master_csv, scan_id, {
                    "feat_id":  feat_id,
                    "feat_path": rel_to_3voxel(root, feat_dst),
                })

                success = True
                shutil.rmtree(work, ignore_errors=True)
                done += 1
                print(f"[#] step6 +1 | scan={scan_id} -> feat={feat_id} | t_step={t1-t0:.1f}s | t_total={seconds_to_hms(time.time()-t0_all)}")
                break

            
            if _is_oom_error(log, rc):
                if attempts >= max_oom:
                    
                    if local_cooldown_sec > 0:
                        local_cooldown[scan_id] = time.time() + local_cooldown_sec
                    print(f"[step6] OOM x{attempts} -> give up in this process (scan={scan_id}); "
                          f"cooldown={local_cooldown_sec}s; see log: {log}")
                    tail_log_print(log, args.log_tail_lines)
                    shutil.rmtree(work, ignore_errors=True)
                    break
                else:
                    
                    print(f"[step6] OOM (attempt {attempts}/{max_oom}) scan={scan_id}  backoff {backoff}s ...")
                    tail_log_print(log, args.log_tail_lines)
                    shutil.rmtree(work, ignore_errors=True)
                    time.sleep(backoff)
                    
            else:
                
                print(f"[step6] FAIL scan={scan_id} rc={rc} (see {log})")
                tail_log_print(log, args.log_tail_lines)
                shutil.rmtree(work, ignore_errors=True)
                break

        
        _release_row_lock(lk)

    print(f"[step6] DONE. finished={done}")





def step7_render(args):
    root = Path(args.root)
    toolkit = Path(args.toolkit)
    blender = args.blender
    master_csv = root / "3_voxel" / "index" / "index.csv"
    ensure_master_initialized(master_csv)
    
    if not master_csv.exists():
        raise SystemExit("[ERROR] Missing index/index.csv; complete Step6 first.")
    
    
    terrain_dirs = sorted([d for d in (root / "1_terrain_splitting").iterdir() if d.is_dir() and not d.name.startswith('.')])

    total_patches = 0
    cached_patches = 0
    patches_to_render = []

    
    for terrain_dir in terrain_dirs:
        patches_dir = terrain_dir / "patches"
        if not patches_dir.exists():
            continue
        for patch_file in patches_dir.glob("patch_*.glb"):
            total_patches += 1
            
            patch_id = f"{terrain_dir.name}_p{patch_file.stem.split('_')[1]}"  
            render_dir = root / "3_voxel" / "renders" / patch_id  
            
            
            if render_dir.exists():
                cached_patches += 1
            else:
                patches_to_render.append((terrain_dir.name, patch_file, patch_id, render_dir))

    
    print(f"Total patches: {total_patches}")
    print(f"Cached patches: {cached_patches}")
    print(f"Patches to render: {len(patches_to_render)}")

    
    start_time = time.time()
    for i, (terrain_tag, patch_file, patch_id, render_dir) in enumerate(patches_to_render, 1):
        remaining = len(patches_to_render) - i
        print(f"Rendering {i}/{total_patches} patches. {remaining} patches remaining.")

        
        
        
        

        
        print(f"Starting rendering for {patch_id}...")
        
        
        ensure_dir(render_dir)

        log_file = root / "3_voxel" / "logs" / "step7" / f"{patch_id}.log"
        ensure_dir(log_file.parent)

        
        cmd = [
            sys.executable, str(toolkit / "render.py"),
            "--input_dir", str(terrain_tag),
            "--output_dir", str(render_dir),  
            "--object_path", str(patch_file.resolve()),
            "--blender", str(blender),
            "--render_subdir", patch_id  
        ]
        
        
        t0 = time.time()
        rc = run_cmd(cmd, log_file, timeout=args.step_timeout, idle_timeout=args.idle_timeout, verbose=args.verbose)
        t1 = time.time()

        
        render_json = render_dir /"renders"/patch_id/ "transforms.json"  
        
        print(f"rc = {rc}")
        if rc != 0 or not render_json.exists():
            print(f"[step7] FAIL patch={patch_id} rc={rc} (see {log_file})")
            tail_log_print(log_file, args.log_tail_lines)
            shutil.rmtree(render_dir, ignore_errors=True)  
            continue  

        
        elapsed_time = t1 - t0
        total_elapsed_time = time.time() - start_time
        print(f"[step7] Rendering successful for {patch_id} | t_step={elapsed_time:.1f}s | t_total={total_elapsed_time:.1f}s")
        print()
        
        
        update_index_with_render_results(master_csv, patch_id, render_dir)
        
        result_dir = render_dir /"renders"/patch_id
        archive_render_results(result_dir)
        

    print(f"[step7] DONE. Rendered {len(patches_to_render)}/{total_patches} patches.")
    
def update_index_with_render_results(index_csv, patch_id, render_dir):
    
    rows = read_csv_rows_locked(index_csv)
    num_patch_id = 0
    for row in rows:
        if row["patch_id"] == patch_id:
            num_patch_id += 1

            render_subdir_id = f"render_{patch_id}"
            row["render_subdir_id"] = render_subdir_id
            row["render_dir"] = f"3_voxel/renders/{render_subdir_id}"

    
    print(f"render_{patch_id}")
    print("!!!!!!!!!!!!!!!1   num_patch_id:",num_patch_id,"       !!!!!!!!!!!!!!!")
    atomic_write_master(index_csv, rows)

def _update_patch_rows(csv_path: Path, patch_id: str, updates: Dict[str, str]) -> int:




    if not csv_path.exists():
        return 0
    rows = read_csv_rows_locked(csv_path)
    changed = 0
    for row in rows:
        if row.get("patch_id") == patch_id:
            for k, v in updates.items():
                if k in row:
                    row[k] = str(v)
            
            changed += 1
    if changed:
        atomic_write_master(csv_path, rows)
    return changed

def archive_render_results(result_dir):



    
    target_dir = result_dir.parent.parent
   
    
    for item in result_dir.iterdir():
        if item.is_dir():
            shutil.move(str(item), str(target_dir / item.name))  
        else:
            shutil.move(str(item), str(target_dir / item.name))  

    
    shutil.rmtree(result_dir, ignore_errors=True)
    shutil.rmtree(result_dir.parent)

def move_contents_and_remove_source(source_path):






    
    target_dir = os.path.dirname(source_path)
    source_dir = source_path
    
    
    if not os.path.exists(source_dir):
        print(f"Error: source directory does not exist - {source_dir}")
        return False
    
    
    if source_dir == target_dir:
        print("Error: source and target directories must differ")
        return False
    
    
    os.makedirs(target_dir, exist_ok=True)
    
    try:
        print(f"Moving: {source_dir} -> {target_dir}")
        
        
        items_moved = 0
        for item in os.listdir(source_dir):
            source_item_path = os.path.join(source_dir, item)
            target_item_path = os.path.join(target_dir, item)
            
            
            if os.path.exists(target_item_path):
                if os.path.isdir(target_item_path):
                    shutil.rmtree(target_item_path)
                else:
                    os.remove(target_item_path)
                print(f"  Overwritten: {item}")
            
            
            shutil.move(source_item_path, target_item_path)
            items_moved += 1
            print(f"  Moved: {item}")
        
        print(f"Moved {items_moved} items")
        
        
        if not os.listdir(source_dir):
            os.rmdir(source_dir)
            print(f"Removed empty directory: {source_dir}")
            return True
        else:
            print(f"Warning: source directory is not empty - {source_dir}")
            return False
            
    except Exception as e:
        print(f"Operation failed: {e}")
        return False

'''
# Legacy render stage.
def step7_render(args):
    root = Path(args.root)
    toolkit = Path(args.toolkit)
    blender = args.blender
    master_csv = root / "3_voxel" / "index" / "index.csv"
    ensure_master_initialized(master_csv) 
    if not master_csv.exists():
        raise SystemExit("[ERROR] Missing index/index.csv; complete Step6 first.")

    py = toolkit / "render.py"
    done = 0; t0_all = time.time()

    while True:
        rows = read_csv_rows_locked(master_csv)
        candidates = [r for r in rows if r.get("feat_id") and not r.get("render_subdir_id")]
        if not candidates:
            if done == 0:
                print("[step7] nothing to do.")
            break

        picked = None
        for r in candidates:
            lk = _lock_path_for(root, "step7", r["scan_id"])
            if _acquire_row_lock(lk, ttl_hours=6.0):
                picked = (r, lk); break
        if not picked:
            time.sleep(0.2); continue

        r, lk = picked
        scan_id = r["scan_id"]
        suffix = scan_id.split("_")[-1] if "_" in scan_id else short_hash(scan_id)
        render_subdir_id = f"render_{suffix}"

        work = root / "temp" / "step7" / suffix
        ensure_dir(work)

        patch_file = Path(r["patch_file"])
        if not patch_file.exists():
            print(f"[step7] WARN patch missing: {patch_file} (skip)")
            _release_row_lock(lk); shutil.rmtree(work, ignore_errors=True); continue

        log = root / "3_voxel" / "logs" / "step7" / f"{render_subdir_id}.log"
        ensure_dir(log.parent)
        cmd = [sys.executable, str(py),
               "--input_dir", str(work),
               "--output_dir", str(work),
               "--object_path", str(patch_file.resolve()),
               "--blender", str(blender),
               "--render_subdir", render_subdir_id]
        tt0 = time.time()
        rc = run_cmd(cmd, log, timeout=args.step_timeout, idle_timeout=args.idle_timeout, verbose=args.verbose)
        tt1 = time.time()

        render_json = work / "renders" / render_subdir_id / "transforms.json"
        if rc != 0 or not render_json.exists():
            print(f"[step7] FAIL scan={scan_id} rc={rc} (see {log})")
            tail_log_print(log, args.log_tail_lines)
            _release_row_lock(lk); shutil.rmtree(work, ignore_errors=True); continue

        # Archive the rendered directory.
        renders_src = work / "renders" / render_subdir_id
        renders_out = root / "3_voxel" / "renders" / render_subdir_id
        if renders_out.exists():
            shutil.rmtree(renders_out, ignore_errors=True)
        shutil.copytree(renders_src, renders_out)

        _update_master_row(master_csv, scan_id, {
            "render_subdir_id": render_subdir_id,
            "render_dir":       rel_to_3voxel(root, renders_out),
        })

        _release_row_lock(lk)
        shutil.rmtree(work, ignore_errors=True)

        done += 1
        print(f"[#] step7 +1 | scan={scan_id} -> render={render_subdir_id} | t_step={tt1-tt0:.1f}s | t_total={seconds_to_hms(time.time()-t0_all)}")

    print(f"[step7] DONE. finished={done}")
'''



def step8_full_feature(args):





    root = Path(args.root)
    toolkit = Path(args.toolkit)
    master_csv = root / "3_voxel" / "index" / "index.csv"
    metadata_csv = root / "3_voxel" / "metadata.csv"

    ensure_master_initialized(master_csv)
    
    ensure_dir(metadata_csv.parent)
    if metadata_csv.exists():
        _ = read_csv_rows_locked(metadata_csv)  
    else:
        atomic_write_master(metadata_csv, [])

    patches = enumerate_patches(root)
    if not patches:
        raise SystemExit("[ERROR] No patches found; complete Step3 and check 1_terrain_splitting/*/patches/.")

    feat_out_dir = root / "3_voxel" / "full_features"
    ensure_dir(feat_out_dir)
    start_time = time.time()
    cached = 0
    success = 0
    oom_failures: List[str] = []   
    other_failures: List[str] = [] 

    print(f"[step8] patches={len(patches)}  output_dir={feat_out_dir}")
    for i, (terrain_tag, pidx, patch_path) in enumerate(patches, 1):
        patch_id = f"{terrain_tag}_p{pidx}"
        full_feat_id = f"full_feat_{patch_id}"
        full_feat_file = feat_out_dir / f"{full_feat_id}.npz"

        
        if full_feat_file.exists():
            cached += 1
            
            updated_master = _update_patch_rows(master_csv, patch_id, {
                "full_feat_id": full_feat_id,
                "full_feat_path": rel_to_3voxel(root, full_feat_file),
            })
            updated_meta = _update_patch_rows(metadata_csv, patch_id, {
                "full_feat_id": full_feat_id,
                "full_feat_path": rel_to_3voxel(root, full_feat_file),
            })
            print(f"[step8] cache hit {full_feat_id} | updated master={updated_master} meta={updated_meta}")
            continue

        work = root / "temp" / "step8" / full_feat_id
        ensure_dir(work)

        log = root / "3_voxel" / "logs" / "step8" / f"{full_feat_id}.log"
        ensure_dir(log.parent)
        cmd = [
            sys.executable, str(toolkit / "extract_full_feature.py"),
            "--input_dir", str(patch_path.parent.parent),
            "--output_dir", str(work),
            "--object_path", str(patch_path.resolve()),
            "--output_name", f"{full_feat_id}.npz",
            "--model", args.step8_model,
            "--terrain_scale", str(args.step8_terrain_scale),
            "--voxel_size", str(args.step8_voxel_size),
        ]

        t0 = time.time()
        rc = run_cmd(cmd, log, timeout=args.step_timeout, idle_timeout=args.idle_timeout, verbose=args.verbose)
        time.sleep(2)
        t1 = time.time()

        src_feat = work / "full_features" / f"{full_feat_id}.npz"
        if rc != 0 or not src_feat.exists():
            if _is_oom_error(log, rc):
                oom_failures.append(patch_id)
                print(f"[step8] OOM ({len(oom_failures)}) patch={patch_id} rc={rc} (see {log})")
            else:
                other_failures.append(patch_id)
                print(f"[step8] FAIL patch={patch_id} rc={rc} (see {log})")
            tail_log_print(log, args.log_tail_lines)
            shutil.rmtree(work, ignore_errors=True)
            continue

        link_or_copy(src_feat, full_feat_file)

        
        updated_master = _update_patch_rows(master_csv, patch_id, {
            "full_feat_id": full_feat_id,
            "full_feat_path": rel_to_3voxel(root, full_feat_file),
        })
        updated_meta = _update_patch_rows(metadata_csv, patch_id, {
            "full_feat_id": full_feat_id,
            "full_feat_path": rel_to_3voxel(root, full_feat_file),
        })

        success += 1
        print(f"[#] step8 {success}/{len(patches)} | patch={patch_id} -> {full_feat_id} "
              f"| t_step={t1-t0:.1f}s | t_total={seconds_to_hms(time.time()-start_time)} "
              f"| updated master={updated_master} meta={updated_meta}")
        shutil.rmtree(work, ignore_errors=True)

    total_fail = len(oom_failures) + len(other_failures)
    elapsed = seconds_to_hms(time.time()-start_time)
    print(f"[step8] DONE.")
    print(f"  success: {success}")
    print(f"  cached:  {cached}")
    print(f"  errors:  {total_fail} (oom={len(oom_failures)}, other={len(other_failures)})")
    if oom_failures:
        print("  OOM patch_id list:")
        for pid in oom_failures:
            print(f"    - {pid}")
    if other_failures:
        print("  Other error patch_id list:")
        for pid in other_failures:
            print(f"    - {pid}")
    print(f"  elapsed: {elapsed}")

def parse_args():
    ap = argparse.ArgumentParser(description="Dataset pipeline (single-process, staged, shard-index for step4)")
    ap.add_argument("--mode", choices=["step1","step2","step3","step4","step5","step6","step7","step8"], required=True)

    ap.add_argument("--blender", required=True, help="Blender executable")
    ap.add_argument("--root", required=True, help="Project root")
    ap.add_argument("--toolkit", required=True, help="dataset_toolkit directory")

    
    ap.add_argument("--aux_raw_dir", default="aux_models/raw")
    ap.add_argument("--begin_aux", type=int, default=0)
    ap.add_argument("--end_aux", type=int, default=159)

    
    ap.add_argument("--terrain_dir", default="terrain_models/offroad",
                    help="Comma-separated directories, e.g. 'terrain_models/offroad,terrain_models/city'")

    
    ap.add_argument("--lidar_height", type=float, default=0.83)
    ap.add_argument("--patch_aspect_max", type=float, default=1.25)
    ap.add_argument("--patch_min_edge", type=float, default=20.0)
    ap.add_argument("--reject_action", choices=["move","delete","keep"], default="move")

    
    ap.add_argument("--target_samples", type=int, default=0, help="Step4 successful-scan target (>0 required)")
    ap.add_argument("--scale_factor", nargs=6, type=float, default=[0.5,10.0,0.5,10.0,0.5,10.0])
    ap.add_argument("--random_pose_rate", type=float, default=0.3)

    
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--log_tail_lines", type=int, default=LOG_TAIL_LINES_DEFAULT)
    ap.add_argument("--step_timeout", type=int, default=200, help="Per-step timeout in seconds (0 disables)")
    ap.add_argument("--idle_timeout", type=int, default=200, help="No-output timeout in seconds (0 disables)")

    
    ap.add_argument("--step6_oom_retries", type=int, default=3,
                    help="Maximum in-process Step6 OOM retries per sample")
    ap.add_argument("--step6_oom_backoff", type=int, default=30,
                    help="Step6 backoff after each OOM in seconds")
    ap.add_argument("--step6_local_cooldown", type=int, default=300,
                    help="Step6 in-process OOM cooldown in seconds (0 disables)")

    
    ap.add_argument("--step8_model", type=str, default="sonata", help="Step8 full feature model name")
    ap.add_argument("--step8_terrain_scale", type=float, default=25.0, help="Step8 terrain scale")
    ap.add_argument("--step8_voxel_size", type=int, default=128, help="Step8 voxel size")

    return ap.parse_args()

def main():
    args = parse_args()
    if args.mode == "step1":
        step1_aux_preproc(args)
    elif args.mode == "step2":
        step2_terrain_preproc(args)
    elif args.mode == "step3":
        step3_terrain_split(args)
    elif args.mode == "step4":
        step4_scan(args)
    elif args.mode == "step5":
        step5_voxel(args)
    elif args.mode == "step6":
        step6_feature(args)
    elif args.mode == "step7":
        step7_render(args)
    elif args.mode == "step8":
        step8_full_feature(args)
    else:
        raise SystemExit("unknown mode")

if __name__ == "__main__":
    main()
