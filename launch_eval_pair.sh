#!/bin/bash 


#list of strings to evaluate
LANG_PAIRS=("hf://datasets/haoranxu/WMT22-Test/cs-en/test-00000-of-00001-1a83a591805d9178.parquet" "hf://datasets/haoranxu/WMT22-Test/de-en/test-00000-of-00001-c03dcec47c23d6ca.parquet" "hf://datasets/haoranxu/WMT22-Test/en-cs/test-00000-of-00001-b92f389a2a10e4b5.parquet" "hf://datasets/haoranxu/WMT22-Test/en-de/test-00000-of-00001-c470e1e53ed73302.parquet" "hf://datasets/haoranxu/WMT22-Test/en-is/test-00000-of-00001-872ab78ba9548351.parquet" "hf://datasets/haoranxu/WMT22-Test/en-ru/test-00000-of-00001-889b8af39e8c83c4.parquet" "hf://datasets/haoranxu/WMT22-Test/en-zh/test-00000-of-00001-6b3b7f42ead58b33.parquet" "hf://datasets/haoranxu/WMT22-Test/en-zh/test-00000-of-00001-6b3b7f42ead58b33.parquet" "hf://datasets/haoranxu/WMT22-Test/en-zh/test-00000-of-00001-6b3b7f42ead58b33.parquet" "hf://datasets/haoranxu/WMT22-Test/is-en/test-00000-of-00001-bb3b8280f4b7ff31.parquet" "hf://datasets/haoranxu/WMT22-Test/ru-en/test-00000-of-00001-4455a1b04d42177e.parquet" "hf://datasets/haoranxu/WMT22-Test/zh-en/test-00000-of-00001-a8c846c3e121c2f6.parquet")

for LANG_PAIR in "${LANG_PAIRS[@]}"
do
    LANG=$(echo "$LANG_PAIR" | awk -F 'hf://datasets/haoranxu/WMT22-Test/' '{print $2}' | awk -F '/' '{print $1}')
    SRC_LANG=$(echo "$LANG" | awk -F '-' '{print $1}')
    TGT_LANG=$(echo "$LANG" | awk -F '-' '{print $2}')
    echo "Evaluating $LANG_PAIR"
    sbatch --job-name "eval-${LANG}" eval_pairs.job --src "${SRC_LANG}" --data_path "${LANG_PAIR}" --tgt "${TGT_LANG}" --dtype "float16" --model "haoranxu/ALMA-7B" --beam 5    
done