#!/bin/bash

# -- Function to extract progress from a given log file --
get_progress() {
  local logfile=$1
  # Grep the last line that has "Processing Commits:" and parse it
  grep "Processing Commits:" "$logfile" | tail -n 1 \
    | sed -nE 's/.*Processing Commits:\s+([0-9]+)%.*\|\s+([0-9]+)\/([0-9]+).*/\1%\ (\2\/\3)/p'
}

# Example usage: update paths according to your environment
BSZZ_PROGRESS=$(get_progress "logs/bszz_output_0_36757.log")
MASZZ_PROGRESS=$(get_progress "logs/maszz_output_0_36757.log")
RSZZ_PROGRESS=$(get_progress "logs/rszz_output_0_36757.log")
LSZZ_PROGRESS=$(get_progress "logs/lszz_output_0_36757.log")

echo "BSZZ progress: ${BSZZ_PROGRESS}"
echo "MASZZ progress: ${MASZZ_PROGRESS}"
echo "RSZZ progress: ${RSZZ_PROGRESS}"
echo "LSZZ progress: ${LSZZ_PROGRESS}"
