# R14 Analyse + Komplett Statusvurdering

## R14 Resultat: 66.2 (#105/244)
- exp=0.271 (høy), surv=0.518
- Deteksjon var nøyaktig: exp=0.294/surv=0.521 vs GT 0.271/0.518

## R14 Bias
| Terreng | KL | Hovedbias |
|---|---|---|
| Forest | 0.188 | Forest +0.143 (overpredikert), Sett -0.046 (underpredikert) |
| Settlement | 0.126 | Sett +0.088 (overpredikert overlevelse) |
| Plains | 0.142 | Empty +0.103 (overpredikert), Sett -0.051 |

## Kritisk funn: Floor 0.01 KOSTER oss 2-3 poeng

**0.01 er en ANBEFALING, ikke en API-regel.** Scoringsdocs sier "Recommendation: Always enforce a minimum probability floor of 0.01". API-validering sjekker kun: ikke-negativ + sum=1.0.

Bevis: R8 brukte floor=0.002 og scoret 81.0 uten feil.

**Floor-kostnad:** ~800 nær-statiske celler × 0.04 ekstra KL ≈ 2-3 poeng.
R6 (floor=0.002) = 68.8. R14 (floor=0.01) = 66.2. Forskjellen (~2.6) forklares delvis av floor.

**Anbefaling:** Tilbake til floor=0.002 for dynamiske celler. Bruk 0.005 som kompromiss.

## DFR (Dead Forest Ratio)
Brukt: 0.316. Faktisk: 0.338. Nær nok (6.5% relativt).

## Profil-ekstrapolering
R14 exp=0.294 er OVER vår høyeste profil (0.265/R6). Ekstrapolering fungerer men er upresist.
Bør legge til R14 som 6. interpolasjonspunkt.

## Sammenligning alle høy-exp runder
| Runde | Exp | Floor | Model | Score |
|---|---|---|---|---|
| R2 | 0.207 | 0.01 | v2 (dårlig) | 49.3 |
| R6 | 0.265 | 0.002 | v6 | 68.8 |
| R11 | 0.296 | 0.002 | v9 (lav prior) | 61.9 |
| R14 | 0.294 | 0.01 | v8-corr | 66.2 |

## Handlingsplan
1. **Tilbake til floor=0.002** (API godtar det, 2-3 poeng gevinst)
2. **Legg til R14 profil** som interpolasjonspunkt for exp=0.27+
3. **Settlement survival-bias:** Bayesian update overboostar overlevelse. Vurder å dempe +4/+2 boosts.
4. **Profiltaket for høy-exp er ~70.** Aksepter at score er lavere for disse rundene.
