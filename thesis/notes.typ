
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