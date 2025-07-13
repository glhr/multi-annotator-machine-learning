#!/bin/bash

# Set the maximum number of allowed jobs (in any state) before submitting a new one
THRESHOLD=80

for script in crowd_hpo_scripts/*.sh; do
    echo "Processing $script"
    while true; do
        # Count the current number of jobs for the user in all states (excluding the header)
        current_jobs=$(squeue -u "$USER" -h -o "%i" | \
        awk -F'[_\\[\\]]' '
          BEGIN { total = 0 }
          {
            if (NF == 1) {
              total++
            } else {
              # Extract the content inside the brackets (e.g. "1-70%70")
              split($3, rangeAndStep, "%")
              split(rangeAndStep[1], startEnd, "-")
              if (length(startEnd) == 2) {
                total += (startEnd[2] - startEnd[1] + 1)
              } else {
                total++
              }
            }
          }
          END { print total }
        ')
        echo "Total number of jobs (including expanded job arrays): $current_jobs"
        if [ "$current_jobs" -ge "$THRESHOLD" ]; then
            echo "Current job count ($current_jobs) is at or above threshold ($THRESHOLD). Waiting 30 seconds..."
            sleep 60
            continue
        fi

        # Try to submit the job
        output=$(sbatch "$script" 2>&1)
        ret=$?
        if [ $ret -eq 0 ]; then
            echo "Submission successful: $output"
            break
        else
            if echo "$output" | grep -q "temporarily unable to accept job"; then
                echo "Slurm is temporarily unavailable for $script. Retrying in 200 seconds..."
                sleep 500
            else
                echo "Error submitting $script: $output"
                break
            fi
        fi
    done
    echo "Waiting 30 seconds..."
    sleep 30
done
