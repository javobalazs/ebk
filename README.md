# ebk

## `prb_save.py`

(Ez volt regen a `prb.py`.)

Egy pelda a model letrehozasara a betanitassal es a lementessel egyutt.

### Letrehozas

```python
import gbUtils as gb
long = gb.readFile("routes_EBK_2_orig_long.csv")
lat = gb.readFile("routes_EBK_2_orig_lat.csv")
mo = gb.model_build(lat, long, 5, testSize=0.0005)
```

### Prediktalas

```python
mlat = (lat)[5454:5459, :-predicted_path]
mlong = (long)[5454:5459, :-predicted_path]
s,f, l,a = mo.ll_predict(mlat, mlong)
```

- `s`: a josolt lat-szegmens.
- `f`: a josolt long-szegmens.

### Mentes

```python
mo.save_model("sandbox/valami")
```

Ez tobb fajlt hoz letre (melyek eleg meretesek), a konkret peldaban:

```txt
valami_descriptor.json
valami_f_10_.json
valami_f_11_.json
valami_f_12_.json
valami_f_8_.json
valami_f_9_.json
valami_s_10_.json
valami_s_11_.json
valami_s_12_.json
valami_s_8_.json
valami_s_9_.json
```

## `prb_load.py`

Ez egy pelda egy, mar kesz model betoltesere,
illetve annak hasznalatara.
Tovabba ujra lementi a modelt mas neven,
es a ket mentes osszehasonlithato (ez utobbira a `sandbox/diff.sh` fajl szolgal).

### Betoltes

```python
# ...
import gbUtils as gb
# ...
mo = gb.load_model("sandbox/valami")
```
