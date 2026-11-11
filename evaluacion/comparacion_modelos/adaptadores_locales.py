"""Empareja por hash las dos ejecuciones locales; nunca exporta filas privadas."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from .evaluador_comun import Registro, comparar


def _csv(ruta: Path) -> list[dict[str, str]]:
    with ruta.open(newline='', encoding='utf-8') as archivo:
        return list(csv.DictReader(archivo))


def _coincide(a: float, b: float) -> bool:
    return math.isclose(a, b, abs_tol=1e-9)


def _huella(ruta: Path) -> str:
    sha = hashlib.sha256()
    with ruta.open('rb') as archivo:
        for bloque in iter(lambda: archivo.read(1024 * 1024), b''):
            sha.update(bloque)
    return sha.hexdigest()


def normalizar_locales(
    manifiesto_resnet: Path,
    predicciones_resnet: Path,
    metadatos_regresion: Path,
    predicciones_regresion: Path,
) -> tuple[list[Registro], list[Registro], dict]:
    """Devuelve solo fotos byte idénticas, con etiquetas e identidades concordantes."""
    filas_manifest = [fila for fila in _csv(manifiesto_resnet)
                      if fila['conjunto'] == 'VALIDACION']
    if len({fila['imagen_id'] for fila in filas_manifest}) != len(filas_manifest):
        raise ValueError('Identificador duplicado en manifiesto ResNet')
    manifest = {fila['imagen_id']: fila for fila in filas_manifest}
    if not manifest:
        raise ValueError('Manifiesto ResNet sin validación')
    resnet = {}
    for fila in _csv(predicciones_resnet):
        imagen = fila['imagen_id']
        if imagen not in manifest:
            raise ValueError('Predicción ResNet ajena al manifiesto')
        meta = manifest[imagen]
        if fila['vaca_id'] != meta['vaca_id'] or not _coincide(float(fila['bcs_real']), float(meta['bcs'])):
            raise ValueError('Identidad o BCS ResNet no coincide con manifiesto')
        ruta_foto = (manifiesto_resnet.parent.parent / meta['ruta_dataset']).resolve()
        if manifiesto_resnet.parent.parent.resolve() not in ruta_foto.parents:
            raise ValueError('Ruta ResNet sale del dataset')
        hash_foto = meta['sha256_procesada']
        if _huella(ruta_foto) != hash_foto:
            raise ValueError('La huella actual de ResNet no coincide con el manifiesto')
        if not hash_foto or hash_foto in resnet:
            raise ValueError('Hash ResNet vacío o duplicado')
        resnet[hash_foto] = Registro('resnet18', hash_foto, meta['vaca_id'],
                                   float(meta['bcs']), float(fila['bcs_predicho']))
    if len(resnet) != len(manifest):
        raise ValueError('Predicciones ResNet incompletas')

    filas_datos = [fila for fila in json.loads(metadatos_regresion.read_text(encoding='utf-8'))
                   if fila['particion'] == 'val' and fila['bcs'] is not None]
    if len({fila['clave'] for fila in filas_datos}) != len(filas_datos):
        raise ValueError('Clave duplicada en metadatos de regresión')
    datos = {fila['clave']: fila for fila in filas_datos}
    if not datos:
        raise ValueError('Metadatos de regresión sin validación')
    regresion = {}
    for fila in _csv(predicciones_regresion):
        clave = fila['clave']
        if clave not in datos:
            raise ValueError('Predicción de regresión ajena a metadatos')
        meta = datos[clave]
        if meta['vaca'] is None or not _coincide(float(fila['real']), float(meta['bcs'])):
            raise ValueError('Identidad o BCS de regresión incompleto')
        hash_foto = meta['sha256']
        if _huella(Path(meta['ruta'])) != hash_foto:
            raise ValueError('La huella actual de regresión no coincide con metadatos')
        if not hash_foto or hash_foto in regresion:
            raise ValueError('Hash de regresión vacío o duplicado')
        valor = fila['pipeline']
        regresion[hash_foto] = Registro('yolo_regresion', hash_foto, str(meta['vaca']),
                                       float(meta['bcs']), float(valor) if valor else None)
    if len(regresion) != len(datos):
        raise ValueError('Predicciones de regresión incompletas')

    compartidos = set(resnet) & set(regresion)
    for clave in compartidos:
        if resnet[clave].vaca != regresion[clave].vaca or not _coincide(resnet[clave].real, regresion[clave].real):
            raise ValueError('Identidad o BCS no coincide entre ejecuciones')
    resumen = {
        'resnet_fotos': len(resnet),
        'regresion_fotos': len(regresion),
        'fotos_hash_iguales': len(compartidos),
        'resnet_sin_par': len(resnet) - len(compartidos),
        'regresion_sin_par': len(regresion) - len(compartidos),
    }
    return ([resnet[k] for k in sorted(compartidos)],
            [regresion[k] for k in sorted(compartidos)], resumen)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifiesto-resnet', type=Path, required=True)
    parser.add_argument('--predicciones-resnet', type=Path, required=True)
    parser.add_argument('--datos-regresion', type=Path, required=True)
    parser.add_argument('--predicciones-regresion', type=Path, required=True)
    a = parser.parse_args()
    resnet, regresion, correspondencia = normalizar_locales(
        a.manifiesto_resnet, a.predicciones_resnet,
        a.datos_regresion, a.predicciones_regresion)
    print(json.dumps({'correspondencia': correspondencia,
                      'comparacion_fotos_byte_identicas': comparar(resnet, regresion)},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
