
# -*- coding: utf-8 -*-
"""
Command‑line interface (CLI) for executing the forecasting pipeline.

Usage:
    python run.py --config config.json

This script loads a configuration file, initializes the logging system,
and triggers the end‑to‑end pipeline defined in `src.pipeline.run_pipeline`.

The purpose of this entry point is to:
1) Provide a clean callable interface from the command line.
2) Ensure that users can easily switch between configuration files.
3) Centralize logging for reproducibility and traceability.
"""
import argparse
from src.pipeline import run_pipeline


from src.logger import get_logger
#initialize logger
log = get_logger("RUN")


if __name__ == '__main__':
    
  
    #parse, command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()
    log.info("Starting pipeline execution.")
    log.info(f"Using configuration file: {args.config}")
    
    #run the pipeline
    out = run_pipeline(args.config)
    print("Pipeline executed. Check the folder output/ for results and CSVs.")
