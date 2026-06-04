
# -*- coding: utf-8 -*-
"""CLI de ejecución del pipeline.
Uso: python run.py --config config.json
"""
import argparse
from src.pipeline import run_pipeline


from src.logger import get_logger
log = get_logger("RUN")


if __name__ == '__main__':
    
  

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.json')
    args = parser.parse_args()
    log.info("Lanzando pipeline")
    log.info(f"Config file: {args.config}")
    out = run_pipeline(args.config)
    print("✅ Pipeline ejecutado. Revisa la carpeta output/ para resultados y CSVs.")
