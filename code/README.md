# ml-seminar/code/

These Matlab/Octave codes rewrite, and mildly extend, those in directory `hh19codes/`.

## example

Run the same classification task neural net training as in HH19, then rerun it using a deterministic optimizer, but starting from a random initial parameter state.  

    $ octave
    >> example1;                         % runtime: a couple of minutes
    >> example1('nm',10000,true,false);  % faster; sometimes finds global min

For more information see

    >> help example1
    >> help netbp2             % like hh19codes/netbp.m
    >> help netopt             % uses Nelder-Mead optimization
