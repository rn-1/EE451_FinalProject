

  2 nodes (original — 1 worker):                                                                                                                                                 
  ./bin/master_realtime_rt --worker-ip 100.116.134.88 --worker-port 9000 --scene random                                                                                          
  # frame split: local rows 0..H/2, worker rows H/2..H                                                                                                                           
                                                                                                                                                                                 
  4 nodes (3 workers):                                                                                                                                                           
  ./bin/master_realtime_rt \                                                                                                                                                     
    --workers "100.116.134.88:9000,100.x.x.x:9000,100.x.x.x:9001" \                                                                                                              
    --scene random --width 1280 --spp 4 --depth 4                                                                                                                                
  # frame split: local rows 0..H/4, workers get H/4..H/2, H/2..3H/4, 3H/4..H                                                                                                     

