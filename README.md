# WinCLIP + Graf Wiedzy RDF

**Student:** Igor Zamojski  
**Implementacja WinCLIP:** https://github.com/caoyunkang/WinClip

---

## Cel projektu

Projekt demonstruje integrację **detekcji anomalii wizualnych** (WinCLIP) z **grafem wiedzy RDF**. Model wykrywa defekty na zdjęciach elementów przemysłowych z zestawu MVTec AD (butelki, siatki), a wynik mapowany jest na węzeł ontologii. Stamtąd - przez zapytania SPARQL - pobierane są przyczyny defektu i zalecane działania naprawcze.

---

## Technologie

- **CLIP** - model wizyjno-językowy mapujący obrazy i tekst do wspólnej przestrzeni osadzeń (cosine similarity)
- **WinCLIP** - detekcja anomalii przez wieloskalowe okno przesuwne + porównanie z galerią dobrych obrazów
- **RDF / Turtle** - ontologia defektów (`knowledge/ontology.ttl`)
- **rdflib** - parsowanie ontologii w Pythonie
- **SPARQL** - zapytania po grafie wiedzy

---

## Instalacja

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

---

## Struktura danych wejściowych

```
data/input/<klasa>/
    good/            ← obrazy referencyjne (bez defektów)
    broken_large/    ← obrazy testowe
    broken_small/
    contamination/
```

---

## Uruchomienie

### Kalibracja progu (train)
```bash
python3 src/train.py bottle
```

### Testowanie (test)
```bash
python3 src/test.py bottle "broken_small/*.png" "broken_large/*.png" "contamination/*.png" "good/*.png"
```

---

## Wyniki

Dla każdego obrazu generowany jest folder `data/output/<podfolder>-<id>/` z:
- `result.json` - wyniki detekcji i klasyfikacji
- `heatmap_overlay.png` - oryginalne zdjęcie wraz z zaznaczoną anomalią

---

## Ontologia (`knowledge/ontology.ttl`)

Prosta hierarchia RDF:

```
ex:DefectType  →  ex:hasCause          →  ex:Cause
               →  ex:recommendedAction →  ex:Action
               →  ex:promptTemplate       (prompt CLIP)
               →  ex:applicableToClass    (np. "bottle", "grid")
```
