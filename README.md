Registrace obrazových sekvencí z experimentálního videooftalmoskopu
==============

- diplomová práca je zložená z dvoch súborov implementovaných v prostredí Python 3.8:
    - video_stabilization.py - registračný algoritmus pre obrazové sekvencie
    - images_stabilization.py - registračný algoritmus pre 2 snímky
    - img_process.py - knižnica s algoritmom 

- pre spustenie algoritmu je nutné do Terminálu zadať príkaz s cestou k vybranej video nahrávke (je možné testovať viac nahrávok na jedno spustenie):
    - python video_stabilization.py images/Study_02_00014_01_R.avi images/Study_02_00014_01_L.avi
- voliteľné parametre pri spustení:
    - voľba referencie pomocou parametru -f:
        python video_stabilization.py -f images/Study_02_00014_01_R.avi = berie 1. snímok videa ako referenciu pre ostatné
        python video_stabilization.py images/Study_02_00014_01_R.avi = berie vždy predchádzajúci snímok ako referenciu

- výsledky:
    - zobrazenie zostávajúceho počtu neregistrovaných snímok - funguje ako odpočítavanie do konca registrácie
    - zobrazenie trvania algoritmu (DURATION)
    - zobrazenie štrukturálnej similarity (MSSIM)
    - zobrazenie odchýlky odmocniny strednej kvadratickej chyby (RMSE)
- vybrané parametre s ich popisom je možné nájsť v priloženej práci
