#!/bin/bash

SNR=-5
python src/LibriSpeech_DNS.py -i /home/data/kbh/LibriSpeech/test-clean -o /home/data/kbh/LibriSpeech_noisy/SNR${SNR} -n /home/data/kbh/DNS-Challenge-16kHz/datasets_fullband/noise_fullband -s ${SNR}
