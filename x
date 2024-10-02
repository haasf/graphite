for i in `seq 0 0.25 1`; do
  for j in `seq 0 .25 1`; do
    for k in `seq 0 25 100`; do
      python3 main.py --tr_lo $i --tr_hi $j --num_xforms_boost $k --num_xforms_mask $k <other parameters>;
    done;
  done;
done
