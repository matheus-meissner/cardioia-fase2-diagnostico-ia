"""CardioIA - Fase 2 | Parte 1: extração de sintomas e sugestão de diagnóstico.

Fluxo:
  1. lê as frases de pacientes em `frases_sintomas.txt` (uma por linha);
  2. lê o mapa de conhecimento `mapa_conhecimento.csv` (Sintoma 1 | Sintoma 2 | Doença Associada);
  3. identifica na frase as expressões de sintomas presentes no mapa;
  4. cruza os sintomas encontrados com as doenças do mapa e ranqueia os diagnósticos.

Uso (a partir desta pasta):
    python extrator_diagnostico.py

AVISO: projeto acadêmico. As sugestões NÃO substituem avaliação médica.
"""
import csv
import re
import sys
import unicodedata
from collections import defaultdict
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
ARQ_FRASES = BASE_DIR / "frases_sintomas.txt"
ARQ_MAPA = BASE_DIR / "mapa_conhecimento.csv"
ARQ_SAIDA = BASE_DIR / "resultado_diagnosticos.csv"

# Gabarito usado apenas para validar o extrator (mesma ordem das linhas do .txt).
GABARITO = [
    "Infarto Agudo do Miocárdio",
    "Angina",
    "Insuficiência Cardíaca",
    "Arritmia Cardíaca",
    "Hipertensão Arterial",
    "Pericardite",
    "Embolia Pulmonar",
    "Trombose Venosa Profunda",
    "Estenose Aórtica",
    "Endocardite Infecciosa",
]


def normalizar(texto: str) -> str:
    """Minúsculas, sem acentos e com espaços normalizados (para comparar expressões)."""
    texto = unicodedata.normalize("NFKD", texto.lower())
    texto = "".join(c for c in texto if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", texto).strip()


def carregar_frases(caminho: Path) -> list[str]:
    with open(caminho, encoding="utf-8-sig") as f:
        return [linha.strip() for linha in f if linha.strip()]


def carregar_mapa(caminho: Path):
    """Retorna (rotulos, linhas, doenca_expressoes).

    rotulos: expressão normalizada -> texto original (para exibição)
    linhas: lista de (expr1, expr2, doença) normalizadas
    doenca_expressoes: doença -> conjunto de expressões normalizadas
    """
    rotulos: dict[str, str] = {}
    linhas: list[tuple[str, str, str]] = []
    doenca_expressoes: dict[str, set[str]] = defaultdict(set)

    with open(caminho, encoding="utf-8-sig", newline="") as f:
        for linha in csv.DictReader(f):
            doenca = linha["Doença Associada"].strip()
            e1, e2 = (normalizar(linha["Sintoma 1"]), normalizar(linha["Sintoma 2"]))
            rotulos.setdefault(e1, linha["Sintoma 1"].strip())
            rotulos.setdefault(e2, linha["Sintoma 2"].strip())
            linhas.append((e1, e2, doenca))
            doenca_expressoes[doenca].update((e1, e2))
    return rotulos, linhas, doenca_expressoes


def extrair_sintomas(frase: str, expressoes) -> list[str]:
    """Encontra as expressões do mapa na frase.

    Se uma expressão está contida em outra mais longa que também foi encontrada
    no mesmo trecho (ex.: "falta de ar" dentro de "falta de ar intensa"), vale
    apenas a mais específica.
    """
    texto = normalizar(frase)
    ocorrencias = []  # (início, fim, expressão)
    for expr in expressoes:
        for m in re.finditer(rf"(?<!\w){re.escape(expr)}(?!\w)", texto):
            ocorrencias.append((m.start(), m.end(), expr))

    def coberta(o):
        return any(
            p[0] <= o[0] and o[1] <= p[1] and (p[1] - p[0]) > (o[1] - o[0])
            for p in ocorrencias
        )

    ordem: dict[str, int] = {}
    for inicio, _, expr in sorted(o for o in ocorrencias if not coberta(o)):
        ordem.setdefault(expr, inicio)
    return list(ordem)


def sugerir_diagnosticos(sintomas: list[str], linhas, doenca_expressoes):
    """Ranqueia doenças por nº de sintomas distintos encontrados.

    Desempate: nº de linhas do mapa em que os DOIS sintomas aparecem juntos.
    Retorna lista de dicts ordenada do mais para o menos provável.
    """
    achados = set(sintomas)
    ranking = []
    for doenca, exprs in doenca_expressoes.items():
        encontrados = [s for s in sintomas if s in exprs]
        if not encontrados:
            continue
        pares = sum(1 for e1, e2, d in linhas if d == doenca and e1 in achados and e2 in achados)
        ranking.append({"doenca": doenca, "sintomas": encontrados, "n": len(encontrados), "pares": pares})
    ranking.sort(key=lambda r: (-r["n"], -r["pares"], r["doenca"]))
    return ranking


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")  # evita erro de acentos no console do Windows
    except AttributeError:
        pass

    frases = carregar_frases(ARQ_FRASES)
    rotulos, linhas, doenca_expressoes = carregar_mapa(ARQ_MAPA)
    print(f"{len(frases)} frases | {len(linhas)} regras no mapa | {len(doenca_expressoes)} doenças\n")

    acertos = 0
    saida = []
    for i, frase in enumerate(frases, start=1):
        sintomas = extrair_sintomas(frase, rotulos)
        ranking = sugerir_diagnosticos(sintomas, linhas, doenca_expressoes)

        print(f"[{i:02d}] {frase}")
        print("     Sintomas identificados:", ", ".join(rotulos[s] for s in sintomas) or "nenhum")
        if not ranking:
            print("     Diagnóstico sugerido: indeterminado (nenhum sintoma do mapa encontrado)\n")
            saida.append([i, frase, "", "indeterminado", ""])
            continue

        top = ranking[0]
        empate = len(ranking) > 1 and (ranking[1]["n"], ranking[1]["pares"]) == (top["n"], top["pares"])
        print(f"     Diagnóstico sugerido: {top['doenca']} ({top['n']} sintomas, {top['pares']} combinações)"
              + ("  [EMPATE - requer avaliação médica]" if empate else ""))
        alternativas = [f"{r['doenca']} ({r['n']})" for r in ranking[1:4]]
        if alternativas:
            print("     Outras hipóteses:", "; ".join(alternativas))
        if i <= len(GABARITO):
            esperado = GABARITO[i - 1]
            ok = top["doenca"] == esperado
            acertos += ok
            print(f"     Gabarito: {esperado} -> {'OK' if ok else 'DIVERGE'}")
        print()
        saida.append([i, frase, "; ".join(rotulos[s] for s in sintomas), top["doenca"],
                      "; ".join(alternativas)])

    print(f"Concordância com o gabarito: {acertos}/{min(len(frases), len(GABARITO))}")

    with open(ARQ_SAIDA, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "frase", "sintomas_identificados", "diagnostico_sugerido", "outras_hipoteses"])
        w.writerows(saida)
    print(f"Resultado salvo em: {ARQ_SAIDA.name}")


if __name__ == "__main__":
    main()
