# Meeting Notes 19 Sept 2025


```
G1

[NODES]
index   0 1 2  3   4 5 6 7   8      9  10
Label   A B C  D   E F G H   I      A  C
flag_   i i i  d   o o d o   o      i  o

ND_ID_  0A  1B 2C 3D 4E 5F 6G 7H 8I  9A 10C

[EDGES]
index   0       1         2
Label   XX      YY        ZZ
Edge    012-3   3-456     69-7810
ED_ID_  0XX     1YY       2ZZ

ED_SIG 0XX-0A_1B_2C-3D  1YY-3D-4E_5F_6G   2ZZ-6G_9A-7H_8I_10C
ED_NumI      3                   1                 2
ED_NumO      1                   3                 3
ED_NDTot     4                   4                 6

found_       1                   0

G2

[NODES]
index   0 1 2  3   4 5 6 7   8      9  10
Label   ! @ #  $   % ^ * (   ~      +   )
flag_   i i i  d   o o d o   o      i  o

ND_ID_  0! 1@ 2# 3$  4% 5^ 6* 7( 8~ 9+ 10)

[EDGES]
index   0      1      2
Label  !!      @@     $$
Edge   3-456   012-3  69-7810
ED_ID_ 0!!     1@@    2$$


ED_SIG    0!!-3$-4%_5^_6*   1@@-0!_1@_2#-3$     2$$-6*_9+-7(_8~_10)
ED_NumI      1                   3                 2
ED_NumO      3                   1                 3
ED_NDTot     4                   4                 6

```

0. ED_NDTot Sorted must be identical.  Parallel Check (1:1)
1. ED_NumI and ED_NumO Sorted must be identical.  Parallel Check (1:1)

Edge Matching
Sort by num of Inputs  create bins  of input locs

2. Check if number of inputs then outputs don't match then edge is not a match.
3. if totals match then check the labels and do a map.







This example shows that we need to use index and label of a node to determine the signature of an edge
