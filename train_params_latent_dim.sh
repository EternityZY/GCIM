for latent in {6,8,10}
do
  for domain in {5,10,20,25}
  do
    echo $latent, $domain, 'latent'$latent+'domain'$domain
    python main.py --latent_dim=$latent --domain_num=$domain --expid='latent'$latent+'domain'$domain --device="cuda:0"
  done
done
