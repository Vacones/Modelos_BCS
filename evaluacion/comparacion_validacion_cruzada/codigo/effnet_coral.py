"""EfficientNet-B0 + CORAL (modelos/con_cajas_ordinal/bcs_ordinal.py, experimento B) con el protocolo común de 66 vacas.

Usa las funciones de esa carpeta sin cambios: recorte de la grupa con 10 % de margen (aumento de 5 a 15 % y
desplazamiento), letterbox a 224, Dropout 0.3, AdamW con wd 1e-4, lote 16, etapa 1 con lr 1e-3 y etapa 2 con
block7 y top a lr 1e-5 (BatchNorm congelada), hasta 30 épocas, paciencia 8 sobre el MAE por vaca de la validación
interna, ReduceLROnPlateau, pesos 1 / fotos por vaca, semilla 20261026 más el pliegue y predicción con la foto y su
espejo. Los niveles son los de esa carpeta (2.25 se une con 2.50).
Lo único que cambia es el reparto: los pliegues vienen de protocolo.tabla y la validación interna de
protocolo.separar_val, para que todos los modelos usen las mismas vacas.
La predicción por foto es 2.50 + 0.25 por el número de umbrales con P(BCS > u_k) > 0.5, y la de la vaca es la
media de sus fotos.

Uso: python effnet_coral.py [REPETICIONES...]   (por defecto de 0 a 9)
"""
import os, sys, time, json
AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, AQUI)
import protocolo as PR  # noqa: E402
os.environ.setdefault("BCS_CARPETA", os.path.dirname(os.path.abspath(PR.ruta_entorno("BCS_DATOS"))))
sys.path.insert(0, os.path.join(PR.RAIZ_REPO, "modelos", "con_cajas_ordinal"))
import numpy as np, pandas as pd  # noqa: E402
import bcs_ordinal as B  # noqa: E402
import keras  # noqa: E402

NOMBRE = "effnet_coral"
CABEZA, SEMILLA = "coral", B.SEMILLAS_B[0]


def correr(rep):
    t0 = time.time()
    df = PR.tabla(rep)
    clases = B.leer_clases(PR.ruta_entorno("BCS_DATOS"))
    tabla = pd.DataFrame({"archivo": df.imagen.values, "vaca": df.cow.values, "bcs": df.bcs.values,
                          "caja_ass": [B.leer_cajas(l, clases) for l in df.label.values]})
    assert tabla.caja_ass.notna().all()
    tabla["nivel"] = B.bcs_a_nivel(tabla.bcs)
    vacas_cv = tabla.groupby("vaca").agg(bcs=("bcs", "first"))
    fuentes = B.preparar_fuentes(tabla, "recorte")
    pred = np.full(len(df), np.nan); info = []
    for p in range(PR.N_PLIEGUES):
        tp = time.time()
        i_ev = np.where(df.pliegue.values == p)[0]; tr = np.where(df.pliegue.values != p)[0]
        i_fit, i_val = PR.separar_val(df, tr, rep)
        x_val = B.matriz([fuentes[i] for i in i_val])
        x_ev = B.matriz([fuentes[i] for i in i_ev]); x_ev_espejo = x_ev[:, :, ::-1, :].copy()
        keras.utils.set_random_seed(SEMILLA + p)
        t_fit = tabla.iloc[i_fit]
        lotes = B.Lotes([fuentes[i] for i in i_fit], B.objetivo(CABEZA, t_fit), B.pesos_por_vaca(t_fit.vaca),
                        aumentar=True, semilla=SEMILLA + p)
        m = B.modelo_completo(CABEZA, t_fit.nivel.to_numpy())
        args = (x_val, tabla.vaca.iloc[i_val].to_numpy(), vacas_cv.bcs)
        h1 = B.entrenar(m, CABEZA, lotes, *args, lr=1e-3, etapa=1)
        B.descongelar_etapa2(m)
        h2 = B.entrenar(m, CABEZA, lotes, *args, lr=B.LR_ETAPA2, etapa=2)
        logits = B._promedio_logits(m, x_ev, x_ev_espejo)
        a = B.acumuladas(CABEZA, logits)
        pred[i_ev] = 2.50 + 0.25 * (a > 0.5).sum(1)
        info.append({"pliegue": p, "epocas_etapa1": len(h1), "epocas_etapa2": len(h2), "segundos": time.time() - tp,
                     "fotos_fit": len(i_fit), "fotos_val": len(i_val), "fotos_test": len(i_ev)})
        print(json.dumps(info[-1]), flush=True)
        keras.backend.clear_session()
    assert not np.isnan(pred).any()
    P = PR.guardar_rep(NOMBRE, rep, df, pred)
    mae = float(np.abs(P - PR.por_vaca(df, pred)[1]).mean())
    json.dump({"rep": rep, "equipo": B.describir_equipo(), "pliegues": info, "segundos_total": time.time() - t0, "mae_vaca": mae},
              open(os.path.join(PR.PARC, f"{NOMBRE}_rep{rep}_log.json"), "w"), indent=1)
    print("repetición", rep, "MAE por vaca", round(mae, 4), "segundos", round(time.time() - t0))


if __name__ == "__main__":
    PR.correr_repeticiones(NOMBRE, correr, [int(a) for a in sys.argv[1:]] or list(range(10)))
