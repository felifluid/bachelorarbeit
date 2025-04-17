
= Fragen

Q: Welches Zitierschema? (Name oder Zahl, Seitenangaben, pro Paragraph oder Satz) \
A: Relativ frei, Paragraph reicht

Q: Wie arbeitet GKW in linearem vs nicht-linearen Fall? \
A: Relativ ähnlich. In beiden Fällen arbeitet GKW mit Fouriertransformierten in #sym.zeta. Nur werden in linearen Fällen die Mischterme verworfen. Jede Mode entwickelt sich dadurch unabhängig voneinander (überlicherweise wird aber sowieso nur eine einzelne Mode berechnet). Im nicht-linearen Fall werden die Daten vor der Ausgabe rücktransformiert und das Potential als 3D-Array ausgegeben, während im linearen Fall die komplexen 2D fourier koeffizienten ausgegeben werden. Für diese Inkonsistenz (?) gibt es keine besonderen Gründe.

Q: Rücktransformation von Hamada zu Toroidalen Koordinaten? (circ / s-#sym.alpha)\
A: Möglich zumindest in Taylor Entwicklung! Could be worth to try tbh.

Q: #sym.zeta Formel in s-#sym.alpha Geometrie fehlerhaft?
$ zeta=-frac(phi,2pi) + s_B s_j abs(q) s $
statt
$ zeta = frac(s_B s_j, 2pi) [2pi abs(q) s - phi] $
A: Yes. Wird im Manual korrigiert. Werde erstmal per Fußnote auf die Unterschiede in der Quelle verweisen.

Q: Quelle für #sym.zeta Formel in CHEASE Geometrie? \
A: gkw/docs/ chease oder etc "CHEASE in GKW"

Q: Kann man bei `gmap` von einer "diskreten Funktion" sprechen? \
A: → ne, eher kontinuierlich Funktion, aber aufgrund numerische Darstellung diskrete Ausgabe. TODO: brauch immernoch einen guten Begriff

Q: Ist der safety factor q in allen Geometrien abhängig von #sym.psi? \
A: Yes. Intern arbeitet GKW zum berechnen mit lokal konstantem q. Aber in all meinen (globalen) Fällen ist q eine Funktion von #sym.psi. Überlicherweise steigt q quadratisch mit #sym.psi. Am Minimum hat es Werte $approx 1$ am Rand Werte von $3~4$

Q: Wie ist #sym.psi in circularer Geometrie definiert? \
A: r/R

Q: Fachbegriff für die "Verschraubung" mit zunehmendem #sym.psi
A: Lokale Magnetische Scherung

Q: Sollte man Bilder im Text zunächst beschreiben? (Was sieht man?)
A: Yes!

Q: Wie übersetzt sich die parallele Randbedingung im linearen Fall für #sym.zeta?
$ hat(f)(psi, k, s) = hat(f)(psi, k, s±1) exp(±i k q) $

Das Potential berechnet sich wie folgt:

$ Phi(psi, s, zeta) = hat(f)(psi, k, s) * exp(i k zeta) + hat(f)^*(psi, k, s) * exp(-i k zeta) $

#sym.zeta ist diesem Fall #sym.zeta\-shift, also eine Funktion von #sym.psi und s. 

Beim betrachten eines Punktes $s_1 > 0.5$ müsste das für #sym.Phi heißen:

$ Phi(psi, s_1, zeta) = hat(f)(psi, k, s_1 - 1) * exp(i k zeta(psi, s_1 - 1) - q) + hat(f)^*(psi, k, s_1 -1) * exp(-i k zeta(psi, s_1 - 1) -q) $

Q: $s$-constant Linien zu Plots hinzufügen?
A: Wenn Zeit ist.
   Hab das mal für die Hamada Koordinaten gemacht. War gar nicht sooo aufwendig. Vielleicht füg ich das noch zu Topovis hinzu.

Q: Ähnlich wie Sophia: Kling-Gupta Efficiency für Interpolation Ergebnisse? (oder ähnliche Form von durchschnittlicher Fehler (#sym.psi) in Abhängigkeit von $s$?)
A: Neee. Is schlecht greifbar - lieber relative Differenz

Q: Definition der Fourier Moden unklar \
  - 2.297: Als Summe auch über $k_psi$
  - A.39: Nur noch über $k_zeta$ → Normiert über Larmor radius $rho_*$
  - ToPoVis: Ohne $k_psi$ - nicht normiert
A:

Q: Warum ist das $s$-Gitter ungleichmäßig in der poloidalen slice? \
  Bei $psi=psi_max$ liegen die $s="const"$ Linien bei $s=plus.minus 0.5$ eng beieinander und bei $s=0$ enger.

  PS: Interessanterweise ist das beim Plot im Büro genau umgekehrt (glaube aber meins ist richtig) 

A:

Q: Wie ist das diskrete Gitter in GKW definiert? \
  Für $psi$ vermutlich
  $ psi_i = (i+1) Delta psi #h(1cm) Delta psi = 1/N_psi #h(1cm) "mit" 0 <= i <= N_psi $
  Bei $s$ denke ich 
  $ s_j = s_0 + j dot Delta s #h(1cm) Delta s = 1/N_s #h(1cm) "mit" 0<=j<=N_s $
  und bei $zeta$
  $ zeta_k = k dot Delta zeta #h(1cm) Delta zeta = L_zeta/N_zeta #h(1cm) L_zeta = 1/n_"min" #h(1cm) 0 <= k <= N_zeta $
  wobei $L_zeta$ der Anteil des Torus ist, der simuliert wurde.

A: Beschränkt auf globale Version

Q: Parallele Randbedingungen in GKW falsch definiert

  Die parallelen Randbedingungen in GKW (A.36) sind nach unseren Ergebnissen falsch und müssten stattdessen
  $ f(psi, zeta, s) = f(psi, zeta minus.plus q, s plus.minus 1) $
  heißen.
  Zumindest nach Florians Herleitung und Ergebnisse aus ToPoVis.
  Die (richtigen) Formeln dazu leite ich übernommen und zitiert aus dem Dokument was Florian erstellt hat in meiner BA her.

A:

Q: Gibt es Vorgaben zum Styling? (Schriftgröße, Seitenabstände, Schriftart, etc.)
A:

Q: Was muss aufs Deckblatt?
A:

