for hidden in {32,64,96}
do
  for gcn in {1,2,3}
  do
    echo $hidden, $gcn, 'hidden'$hidden+'gcn'$gcn
    python main.py --hidden_dim=$hidden --gcn_depth=$gcn --expid='hidden'$hidden+'gcn'$gcn --device="cuda:1"
  done
done
