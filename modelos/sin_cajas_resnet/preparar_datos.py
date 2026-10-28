from pathlib import Path
from io import BytesIO
from collections import defaultdict
import csv
import hashlib
import json

from PIL import Image, ImageOps
import pillow_heif

pillow_heif.register_heif_opener()
BASE = Path(__file__).resolve().parent
ORIGINALES = BASE / "datos" / "originales"
PROCESADAS = BASE / "datos" / "procesadas"
REPORTES = BASE / "reportes"
EXTENSIONES = {".jpg", ".jpeg", ".png", ".heic", ".heif"}
EXCLUIDAS = {"basurero", "vaca extra", "vacas fotos", "todas"}

# Lado máximo para una copia inicial; conserva proporciones.
# Se puede ajustar antes de ejecutar para el modelo que utilizarán.
LADO_MAXIMO = 1024


def escribir_csv(ruta, filas, columnas):
    with ruta.open("w", newline="", encoding="utf-8") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=columnas)
        escritor.writeheader()
        escritor.writerows(filas)


if not ORIGINALES.is_dir():
    raise SystemExit(f"No existe: {ORIGINALES}")
PROCESADAS.mkdir(parents=True, exist_ok=True)
REPORTES.mkdir(parents=True, exist_ok=True)

# Conserva etiquetas ya introducidas al repetir la preparación.
previas = {}
inventario = BASE / "inventario.csv"
if inventario.exists():
    with inventario.open(newline="", encoding="utf-8") as archivo:
        for fila in csv.DictReader(archivo):
            previas[fila["imagen_id"]] = fila

filas, errores, ignorados = [], [], []
por_hash = defaultdict(list)
candidatas = 0

for ruta in sorted(ORIGINALES.rglob("*")):
    if not ruta.is_file():
        continue
    relativa = ruta.relative_to(ORIGINALES)
    # Omite metadatos de Finder y de archivos ZIP de macOS.
    if any(p.startswith(".") or p == "__MACOSX" for p in relativa.parts):
        continue
    if any(p.strip().lower() in EXCLUIDAS for p in relativa.parts[:-1]):
        ignorados.append({"archivo": relativa.as_posix(),
                          "motivo": "carpeta excluida"})
        continue
    if ruta.suffix.lower() not in EXTENSIONES:
        ignorados.append({"archivo": relativa.as_posix(),
                          "motivo": "extensión no seleccionada"})
        continue
    candidatas += 1
    try:
        if len(relativa.parts) != 2:
            raise ValueError("Se espera originales/ID_vaca/imagen")
        vaca_id = relativa.parts[0]
        imagen_id = hashlib.sha256(
            relativa.as_posix().encode("utf-8")
        ).hexdigest()
        datos = ruta.read_bytes()
        hash_original = hashlib.sha256(datos).hexdigest()
        with Image.open(BytesIO(datos)) as original:
            imagen = ImageOps.exif_transpose(original).convert("RGB")
            imagen.load()
        ancho_original, alto_original = imagen.size
        imagen.thumbnail((LADO_MAXIMO, LADO_MAXIMO))
        salida = PROCESADAS / vaca_id / f"{imagen_id}.jpg"
        salida.parent.mkdir(parents=True, exist_ok=True)
        imagen.save(salida, "JPEG", quality=95)
        previa = previas.get(imagen_id, {})
        filas.append({
            "imagen_id": imagen_id,
            "vaca_id": vaca_id,
            "ruta_original": ruta.relative_to(BASE).as_posix(),
            "ruta_imagen": salida.relative_to(BASE).as_posix(),
            "ancho_original": ancho_original,
            "alto_original": alto_original,
            "ancho": imagen.width,
            "alto": imagen.height,
            "sha256_original": hash_original,
            "bcs": previa.get("bcs", ""),
            "conjunto": previa.get("conjunto", ""),
        })
        por_hash[hash_original].append(relativa.as_posix())
        print(f"OK: {relativa}")
    except Exception as error:
        errores.append({"archivo": relativa.as_posix(), "error": str(error)})

columnas = ["imagen_id", "vaca_id", "ruta_original", "ruta_imagen",
            "ancho_original", "alto_original", "ancho", "alto",
            "sha256_original", "bcs", "conjunto"]
escribir_csv(inventario, filas, columnas)
escribir_csv(REPORTES / "errores.csv", errores, ["archivo", "error"])
escribir_csv(REPORTES / "ignorados.csv", ignorados, ["archivo", "motivo"])

# Solo detecta originales exactamente iguales byte a byte.
duplicados = []
for firma, archivos in por_hash.items():
    if len(archivos) > 1:
        for archivo in archivos:
            duplicados.append({"sha256_original": firma, "archivo": archivo})
escribir_csv(REPORTES / "duplicados.csv", duplicados,
             ["sha256_original", "archivo"])

conteos = defaultdict(int)
for fila in filas:
    conteos[fila["vaca_id"]] += 1
escribir_csv(REPORTES / "fotos_por_vaca.csv",
             [{"vaca_id": vaca, "fotos": n}
              for vaca, n in sorted(conteos.items())],
             ["vaca_id", "fotos"])
resumen = {
    "imagenes_candidatas": candidatas,
    "imagenes_preparadas": len(filas),
    "errores": len(errores),
    "archivos_ignorados": len(ignorados),
    "vacas_distintas": len(conteos),
    "grupos_de_duplicados_exactos": sum(
        len(archivos) > 1 for archivos in por_hash.values()
    ),
    "imagenes_sin_bcs": sum(not f["bcs"].strip() for f in filas),
}
(REPORTES / "resumen.json").write_text(
    json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8"
)
print(json.dumps(resumen, indent=2, ensure_ascii=False))
if errores:
    raise SystemExit("Revisar reportes/errores.csv antes de continuar")
