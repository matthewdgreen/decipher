#!/usr/bin/env bash
# Resumable per-case driver for the frontier suite on the bundled Zenith model.
set -u
cd /sessions/lucid-loving-ptolemy/mnt/decipher
OUT=/sessions/lucid-loving-ptolemy/mnt/outputs/frontier_zenith_rerun
mkdir -p "$OUT/percase"

IDS=(
 synth_en_40wb_s1 synth_en_80nb_s1 parity_fr_ss_synth_001 parity_it_ss_synth_001
 parity_tool_zenith_goldbug synth_en_150honb_s1 synth_en_80honb_s2 parity_tool_zenith_zodiac408
 synth_en_120vignb_s21 synth_en_120bfnb_s22 synth_en_120vbfnb_s23 synth_en_120grnb_s24
 kryptos_k1_keyed_vigenere kryptos_k2_keyed_vigenere kryptos_k3_transmatrix
 synth_en_120thonb_reverse_s12 synth_en_180thonb_ndown_s15 synth_en_120thonb_hidden_route_s16
 borg_single_B_borg_0171v borg_single_B_borg_0109v copiale_single_B_copiale_p068
)

CALL_BUDGET=${1:-38}   # seconds of wall budget for this invocation
start=$(date +%s)
for tid in "${IDS[@]}"; do
  done_file="$OUT/percase/${tid}.jsonl"
  # skip if already has a completed row or is marked over-limit
  if [ -s "$done_file" ]; then continue; fi
  if [ -f "$OUT/percase/${tid}.skip" ]; then continue; fi
  # stop if we're near the call budget
  elapsed=$(( $(date +%s) - start ))
  if [ "$elapsed" -ge "$CALL_BUDGET" ]; then echo "BUDGET_REACHED after $elapsed s"; break; fi
  remain=$(( CALL_BUDGET - elapsed ))
  echo ">>> running $tid (remain ${remain}s)"
  DECIPHER_PARALLEL_WORKERS=4 DECIPHER_ZENITH_NATIVE_ENGINE=python DECIPHER_TRANSFORM_RANK_ENGINE=python \
  DECIPHER_QUAGMIRE_ENGINE=python DECIPHER_NULL_MASK_ENGINE=python_reference \
  DECIPHER_NGRAM_MODEL_EN=models/ngram5_en_zenith.bin DECIPHER_HOMOPHONIC_SCORE_PROFILE=zenith_native \
  PYTHONPATH=src timeout "$remain" python3 scripts/run_frontier_suite.py \
    --suite-file frontier/automated_solver_frontier.jsonl --solvers decipher \
    --test-id "$tid" --artifact-dir "$OUT/percase/art" \
    --summary-jsonl "$OUT/percase/${tid}.tmp.jsonl" --summary-csv "$OUT/percase/${tid}.tmp.csv" 2>&1 \
    | grep -E "\-> (completed|failed)"
  # only commit if a row was actually produced (case fully ran)
  if [ -s "$OUT/percase/${tid}.tmp.jsonl" ]; then
    mv "$OUT/percase/${tid}.tmp.jsonl" "$done_file"
    mv "$OUT/percase/${tid}.tmp.csv" "$OUT/percase/${tid}.csv" 2>/dev/null
  else
    echo "!!! $tid did not finish in ${remain}s (no row)"; rm -f "$OUT/percase/${tid}.tmp.jsonl"
    # if it got a near-full window and still failed, mark over-limit so we stop retrying
    if [ "$remain" -ge 40 ]; then echo "over-limit" > "$OUT/percase/${tid}.skip"; fi
  fi
done
echo "=== STATUS ==="
have=$(ls "$OUT/percase"/*.jsonl 2>/dev/null | grep -v tmp | wc -l)
echo "completed: $have / ${#IDS[@]}"
