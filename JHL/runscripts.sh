#'chmod u+x runscripts.sh' to use

for i in {0..3}
do
  python3 inference_ln.py $i
  python3 inference_dir.py $i
  python3 inference_ln_shuffle.py $i
  python3 inference_dir_shuffle.py $i
done
