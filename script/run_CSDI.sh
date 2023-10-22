#!/bin/sh

#$ -cwd

export CUDA_VISIBLE_DEVICES="0"

cd /path/to/directory
LOGPATH=../../logs

MISSING_RATE="0.1"

LOG="${LOGPATH}/CSDI_MR-${MISSING_RATE}.log"
python3 exe.py \
    --testmissingratio $MISSING_RATE \
    --batch_size 8 \
    --epochs 3000 \
    --inputdir ../../indata/data_csdi | tee $LOG

MISSING_RATE="0.2"

LOG="${LOGPATH}/CSDI_MR-${MISSING_RATE}.log"
python3 exe.py \
    --testmissingratio $MISSING_RATE \
    --batch_size 8 \
    --epochs 3000 \
    --inputdir ../../indata/data_csdi | tee $LOG

MISSING_RATE="0.3"

LOG="${LOGPATH}/CSDI_MR-${MISSING_RATE}.log"
python3 exe.py \
    --testmissingratio $MISSING_RATE \
    --batch_size 8 \
    --epochs 3000 \
    --inputdir ../../indata/data_csdi | tee $LOG


