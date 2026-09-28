"""
Deja el .docx sin nada que no sea el manuscrito.

Que quita
---------
1. `word/comments.xml` y su relacion. La libreria la escribe siempre, aunque
   el documento no tenga ni un comentario. Word no muestra nada, pero la parte
   queda ahi y cualquier inspector de metadatos la reporta.
2. Propiedades personalizadas vacias.
3. Rellena titulo, autor, materia e idioma en las propiedades del documento,
   que por defecto salian como "Un-named".

Que NO quita
------------
`footnotes.xml` y `endnotes.xml` vacios: Word los espera y quitarlos hace que
algunas versiones marquen el archivo como danado.

Uso:  python3 scripts/limpiar_docx.py manuscrito_GRL.docx
"""
import pathlib
import re
import shutil
import sys
import zipfile

TITULO = ("A Second-Order Damage Tensor Is Sufficient for Damage-Induced "
          "Permeability; Scalar Crack Density Is Not")
AUTOR = ("German Orlando Romero Suárez; Diego Fernando Villegas Bermúdez; "
         "Wilmer Velilla Díaz")
MATERIA = "Damage-permeability closures; phase-field fracture; FFT homogenization"

FUERA = {"word/comments.xml", "word/_rels/comments.xml.rels"}

CORE = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<cp:coreProperties '
        'xmlns:cp="http://schemas.openxmlformats.org/package/2006/metadata/core-properties" '
        'xmlns:dc="http://purl.org/dc/elements/1.1/" '
        'xmlns:dcterms="http://purl.org/dc/terms/" '
        'xmlns:dcmitype="http://purl.org/dc/dcmitype/" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        f'<dc:title>{TITULO}</dc:title>'
        f'<dc:subject>{MATERIA}</dc:subject>'
        f'<dc:creator>{AUTOR}</dc:creator>'
        f'<cp:lastModifiedBy>{AUTOR}</cp:lastModifiedBy>'
        '<cp:revision>1</cp:revision>'
        '<dc:language>en-US</dc:language>'
        '</cp:coreProperties>')


def limpiar(ruta):
    ruta = pathlib.Path(ruta)
    tmp = ruta.with_suffix(".tmp.docx")
    quitados = []
    with zipfile.ZipFile(ruta) as z, zipfile.ZipFile(
            tmp, "w", zipfile.ZIP_DEFLATED) as out:
        for item in z.infolist():
            if item.filename in FUERA:
                quitados.append(item.filename)
                continue
            datos = z.read(item.filename)
            if item.filename == "[Content_Types].xml":
                datos = re.sub(
                    rb'<Override PartName="/word/comments.xml"[^>]*/>', b"",
                    datos)
            elif item.filename == "word/_rels/document.xml.rels":
                datos = re.sub(
                    rb'<Relationship[^>]*Target="comments.xml"[^>]*/>', b"",
                    datos)
            elif item.filename == "docProps/core.xml":
                datos = CORE.encode("utf-8")
            out.writestr(item, datos)
    shutil.move(tmp, ruta)
    return quitados


def revisar(ruta):
    """Chequeo final: lo que un inspector de metadatos miraria."""
    z = zipfile.ZipFile(ruta)
    x = z.read("word/document.xml").decode()
    pruebas = {
        "marca de agua": r"WaterMark|watermark|w:pict|v:shape",
        "fondo de pagina": r"<w:background",
        "control de cambios": r"w:ins |w:del |trackChanges",
        "comentarios": r"commentRangeStart|commentReference",
        "resaltado": r"w:highlight",
        "campos con codigo": r"<w:fldChar",
        "texto entre corchetes": r"\[(?!u,|1 \+|tr|K)",
    }
    print(f"  {'prueba':24s} resultado")
    ok = True
    for nombre, patron in pruebas.items():
        n = len(re.findall(patron, x))
        ok &= n == 0
        print(f"  {nombre:24s} {'limpio' if n == 0 else str(n) + ' casos'}")
    partes = [n for n in z.namelist() if not n.endswith("/")]
    print(f"  {'partes del paquete':24s} {len(partes)}")
    print(f"  {'comments.xml':24s} "
          f"{'ausente' if 'word/comments.xml' not in partes else 'PRESENTE'}")
    core = z.read("docProps/core.xml").decode()
    autor = re.search(r"<dc:creator>(.*?)</dc:creator>", core)
    print(f"  {'autor en propiedades':24s} {autor.group(1) if autor else '—'}")
    return ok


if __name__ == "__main__":
    archivo = sys.argv[1] if len(sys.argv) > 1 else "manuscrito_GRL.docx"
    print("quitado:", limpiar(archivo) or "nada")
    print()
    revisar(archivo)
