Exercitiul 1:
Am incarcat setul de date 'bike_daily.csv' si am vizualizat relatiile dintre variabile. Am folosit grafice de tip scatter (puncte) pentru a vedea cum influenteaza temperatura, umiditatea si vantul numarul de inchirieri. Am observat ca temperatura are o relatie curbilinie (neliniara) cu numarul de biciclete inchiriate.

Exercitiul 2:
Am standardizat datele (scazand media si impartind la deviatia standard) pentru a ajuta algoritmul PyMC sa functioneze mai bine. Am construit doua modele probabiliste:
1. Un model Liniar simplu.
2. Un model Polinomial in care am adaugat patratul temperaturii (temp^2) pentru a surprinde curbura observata la exercitiul 1.

Exercitiul 3:
Am rulat algoritmul de esantionare (sampling) pentru ambele modele pentru a gasi parametrii (beta). Am folosit 'az.summary' si 'trace plots' pentru a verifica daca lanturile Markov au converis corect si pentru a vedea care variabila are coeficientul cel mai mare (influenta cea mai puternica).

Exercitiul 4:
Am folosit criteriul WAIC pentru a compara cele doua modele. Modelul Polinomial a avut un scor mai bun (WAIC mai mic), deci se potriveste mai bine pe date. Am facut un "Posterior Predictive Check" (PPC), desenand linia de predictie a modelului si intervalul de incertitudine (HDI) peste datele reale, aratand ca modelul urmareste corect curbura datelor.

Exercitiul 5:
Am transformat problema intr-una de clasificare. Am creat o variabila noua binara 'is_high_demand' (1 daca numarul de inchirieri e in top 25%, 0 in caz contrar). Am construit un model de Regresie Logistica in PyMC folosind functia sigmoid (Bernoulli likelihood) pentru a prezice probabilitatea de cerere mare.

Exercitiul 7:
Am extras intervalele de densitate inalta (95% HDI) pentru coeficientii modelului logistic. Am analizat acesti coeficienti pentru a determina care factor (temperatura, vacanta, etc.) creste cel mai mult probabilitatea ca o zi sa fie "high demand".
