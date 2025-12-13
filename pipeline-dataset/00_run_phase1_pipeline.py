#!/usr/bin/env python3
"""
PHASE 1: K-BERT Spanish NER Pipeline - Complete Execution
Working directory: /home/omar/projects/K-BERT/pipeline-dataset/
Output: ./outputs/

CORRECTED: Uses only Python standard library - NO numpy, pandas, or external dependencies
"""

import os
import sys
import time
import subprocess
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

WORK_DIR = Path.cwd()
print(f"Working directory: {WORK_DIR}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('phase1_pipeline_complete.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_python_script(script_path: Path, description: str) -> Tuple[bool, str]:
    """Runs a Python script and captures output."""
    logger.info(f"\n{'=' * 80}")
    logger.info(f"EXECUTING: {description}")
    logger.info(f"Script: {script_path.name}")
    logger.info(f"Working dir: {WORK_DIR}")
    logger.info(f"{'=' * 80}")
    
    try:
        start_time = time.time()
        
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=3600,
            cwd=WORK_DIR
        )
        
        elapsed_time = time.time() - start_time
        
        if result.stdout:
            logger.info(result.stdout)
        
        if result.returncode != 0:
            logger.error(f"Script failed with return code {result.returncode}")
            if result.stderr:
                logger.error(f"Error output:\n{result.stderr}")
            return False, result.stderr
        
        logger.info(f"✓ Completed in {elapsed_time:.2f} seconds")
        return True, ""
    
    except subprocess.TimeoutExpired:
        logger.error(f"Script timeout after 1 hour")
        return False, "Script timeout"
    except Exception as e:
        logger.error(f"Error running script: {str(e)}")
        return False, str(e)

def main():
    logger.info("=" * 80)
    logger.info("PHASE 1: K-BERT Spanish NER Pipeline - Complete Execution")
    logger.info("=" * 80)
    logger.info(f"Start time: {datetime.now()}")
    
    pipeline_start = time.time()
    
    steps = [
        (WORK_DIR / '01_download_conll2002.py', 'Download and Validate CoNLL 2002 Spanish Dataset'),
        (WORK_DIR / '02_format_for_kbert.py', 'Format Dataset for K-BERT Training'),
        (WORK_DIR / '03_prepare_splits.py', 'Prepare Training Splits & BETO Compatibility Check'),
    ]
    
    all_success = True
    for script_path, description in steps:
        if not script_path.exists():
            logger.error(f"Script not found: {script_path}")
            all_success = False
            continue
        
        success, error_msg = run_python_script(script_path, description)
        
        if not success:
            all_success = False
            logger.error(f"Pipeline failed at step: {description}")
            break
    
    pipeline_duration = time.time() - pipeline_start
    
    logger.info("\n" + "=" * 80)
    logger.info("PHASE 1 EXECUTION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total pipeline duration: {pipeline_duration:.2f} seconds")
    
    if all_success:
        logger.info("\n🎉 ✓ PHASE 1 COMPLETE - READY FOR K-BERT TRAINING 🎉\n")
        logger.info("Output location: ./outputs/conll2002_training_ready/")
        logger.info("Next step: Update BETO model path and start training\n")
        return True
    else:
        logger.error("\n✗ PHASE 1 FAILED - CHECK ERRORS ABOVE ✗\n")
        return False

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}", exc_info=True)
        exit(1)
