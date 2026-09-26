#!/bin/bash
# Appends an E6 progress snapshot every 10 minutes to e6_progress.log.
# Start:  nohup bash scripts/watch_e6.sh > /dev/null 2>&1 &
# Stop :  pkill -f watch_e6.sh
REPO="/scratch/user/mohamed.shaaban/mohamed_shaaban_29_20260916_181506/OpenFedLLM"
OUTROOT="/scratch/user/mohamed.shaaban/mohamed_shaaban_29_20260916_181506/non_fl_lowres"
LOG="${REPO}/e6_progress.log"
cd "$REPO"

while true; do
  {
    echo "════════════════════════════════════════════  $(date '+%Y-%m-%d %H:%M:%S')"
    Q=$(squeue -u mohamed.shaaban -h -o "%.10i %.12j %.8T %.10M %R" 2>/dev/null)
    if [ -n "$Q" ]; then echo "QUEUE:"; echo "$Q" | sed 's/^/  /'
    else echo "QUEUE: (no jobs — finished, failed, or not submitted)"; fi

    echo "RUNS:"
    for d in "$OUTROOT"/*/; do
      [ -d "$d" ] || continue
      n=$(basename "$d")
      ck=$(ls -d "$d"checkpoint-* 2>/dev/null | sed 's/.*checkpoint-//' | sort -n | tail -1)
      fin=$(compgen -G "$d/adapter_model*" > /dev/null && echo yes || echo no)
      sz=$(du -sh "$d" 2>/dev/null | cut -f1)
      # last reported step/loss from the trainer's own state file
      st=$(ls -t "$d"checkpoint-*/trainer_state.json 2>/dev/null | head -1)
      prog=""
      if [ -n "$st" ]; then
        prog=$(python3 -c "
import json,sys
d=json.load(open('$st'))
h=d.get('log_history') or [{}]
print(f\"step {d.get('global_step','?')}/{d.get('max_steps','?')} ep {d.get('epoch',0):.2f} loss {h[-1].get('loss','?')}\")
" 2>/dev/null)
      fi
      printf "  %-34s ckpt=%-6s final=%-3s %-6s %s\n" "$n" "${ck:--}" "$fin" "$sz" "$prog"
    done

    echo "ERRORS (last 2 lines of newest .err):"
    e=$(ls -t logs/e6_lowres_*.err 2>/dev/null | head -1)
    if [ -n "$e" ] && [ -s "$e" ]; then tail -2 "$e" | sed 's/^/  /'; else echo "  (none)"; fi
    echo
  } >> "$LOG" 2>&1
  sleep 600
done
