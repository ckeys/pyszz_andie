#!/bin/bash

# -- Function to extract progress from a given log file --
get_progress() {
  local logfile=$1
  # Grep the last line that has "Processing Commits:" and parse it
  grep "Processing Commits:" "$logfile" | tail -n 1 \
    | sed -nE 's/.*Processing Commits:\s+([0-9]+)%.*\|\s+([0-9]+)\/([0-9]+).*/\1%\ (\2\/\3)/p'
}

# Example usage: update paths according to your environment
BSZZ_PROGRESS=$(get_progress "logs/bszz_output_0_4912.log")
MASZZ_PROGRESS=$(get_progress "logs/maszz_output_0_4912.log")
RSZZ_PROGRESS=$(get_progress "logs/rszz_output_0_4912.log")
LSZZ_PROGRESS=$(get_progress "logs/lszz_output_0_4912.log")

echo "bitcoin BSZZ progress: ${BSZZ_PROGRESS}"
echo "bitcoin MASZZ progress: ${MASZZ_PROGRESS}"
echo "bitcoin RSZZ progress: ${RSZZ_PROGRESS}"
echo "bitcoin LSZZ progress: ${LSZZ_PROGRESS}"


PROJECT_NAME="spring-boot"
BSZZ_PROGRESS=$(get_progress "logs/${PROJECT_NAME}_bszz_output_0_4033.log")
MASZZ_PROGRESS=$(get_progress "logs/${PROJECT_NAME}_maszz_output_0_4033.log")
RSZZ_PROGRESS=$(get_progress "logs/${PROJECT_NAME}_rszz_output_0_4033.log")
LSZZ_PROGRESS=$(get_progress "logs/${PROJECT_NAME}_lszz_output_0_4033.log")

echo "${PROJECT_NAME} BSZZ progress: ${BSZZ_PROGRESS}"
echo "${PROJECT_NAME} MASZZ progress: ${MASZZ_PROGRESS}"
echo "${PROJECT_NAME} RSZZ progress: ${RSZZ_PROGRESS}"
echo "${PROJECT_NAME} LSZZ progress: ${LSZZ_PROGRESS}"

PROJECT_NAME="spring-framework"
BSZZ_PROGRESS=$(get_progress "logs/${PROJECT_NAME}_bszz_output_0_3158.log")
MASZZ_PROGRESS=$(get_progress "logs/${PROJECT_NAME}_maszz_output_0_3158.log")
RSZZ_PROGRESS=$(get_progress "logs/${PROJECT_NAME}_rszz_output_0_3158.log")
LSZZ_PROGRESS=$(get_progress "logs/${PROJECT_NAME}_lszz_output_0_3158.log")

echo "${PROJECT_NAME} BSZZ progress: ${BSZZ_PROGRESS}"
echo "${PROJECT_NAME} MASZZ progress: ${MASZZ_PROGRESS}"
echo "${PROJECT_NAME} RSZZ progress: ${RSZZ_PROGRESS}"
echo "${PROJECT_NAME} LSZZ progress: ${LSZZ_PROGRESS}"
