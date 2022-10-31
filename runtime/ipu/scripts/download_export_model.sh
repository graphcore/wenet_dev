

if [ ! -e "20210618_u2pp_conformer_exp.tar.gz" ]
then
    wget -c http://mobvoi-speech-public.ufile.ucloud.cn/public/wenet/aishell2/20210618_u2pp_conformer_exp.tar.gz;
    tar -xzvf 20210618_u2pp_conformer_exp.tar.gz;
else
    echo "dowload ckpt found"
    echo "skipping downloading and extract"
fi

# NOTE(kan-bayashi): Use UTF-8 in Python to avoid UnicodeDecodeError when LC_ALL=C
export PYTHONIOENCODING=UTF-8
export PYTHONPATH=../../:$PYTHONPATH



# download and untar the model



python3 ../../wenet/bin/export_popef_ipu.py run \
        --ckpt_folder="20210618_u2pp_conformer_exp" \
        --ckpt_file="final.pt" \
        --cmvn_file="global_cmvn" \
        --vocab_file="words.txt" \
        --output_folder="exported_popef"
