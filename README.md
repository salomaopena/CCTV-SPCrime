# CCTV-SPCrime

**Conjunto de dados de imagens de vigilância anotado segundo uma política orientada a eventos, para a detecção de incidentes de segurança pública.**

[![DOI](https://img.shields.io/badge/DOI-A%20PREENCHER-blue)](./CITATION.cff) [![License](https://img.shields.io/badge/license-CC%20BY%204.0-orange)](./LICENSE) [![Version](https://img.shields.io/badge/version-1.0.0-informational)](./CHANGELOG.md)

> **Documentação:** este repositório segue normas internacionais de documentação de dados — ver o [DATASHEET.md](DATASHEET.md) (Datasheets for Datasets, Gebru et al.) e o [CITATION.cff](CITATION.cff). O conjunto adere aos princípios **FAIR** (Findable, Accessible, Interoperable, Reusable).

---

## Visão geral

O **CCTV-SPCrime** é um conjunto de dados de imagens estáticas extraídas de cenários de videovigilância, anotadas para **oito classes de incidentes de segurança pública** segundo uma **política de anotação orientada a eventos** — na qual o incidente, e não o objeto isolado, é a unidade de anotação. Destina-se à **avaliação reprodutível** de modelos de detecção de incidentes.

- **Modalidade:** imagens (frames) 640×640 px, rótulos no formato YOLO + atributos de evento.
- **Classes (8):** acidente, comportamento suspeito, crime, incêndio, intrusão, objeto suspeito, queda, vandalismo.
- **Divisão:** 80% treino / 10% validação / 10% teste.
- **Total:** : 4.042 imagens.
- **Licença:** CC BY 4.0.
- **Trabalhos associados:** data paper (Pena et al., 2026) e arquitetura AIVIS.GCUB (Pena et al., 2026) — ver [Citação](#citação).

## Classes e distribuição

| Classe | Descrição resumida | Treino | Validação | Teste | Total |
|---|---|---|---|---|---|
| accident | colisões, quedas em massa, emergências médicas visíveis | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` |
| suspicious_behavior | movimentos atípicos, fuga, perseguição | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` |
| crime | violência física ou subtração de bens (com atributos) | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` |
| fire | fogo, fumaça, chamas | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` |
| intrusion | transposição de perímetro/áreas proibidas | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` |
| suspicious_object | objetos abandonados, armas | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` |
| fall | pessoa caída em postura anômala | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` |
| vandalism | danos à propriedade | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` |

> Os números devem ser preenchidos após a recontagem (ver [Proveniência e licenças](#proveniência-e-licenças)).

## Estrutura do repositório

``` 
CCTV-SPCrime/
├── README.md                 # este arquivo
├── DATASHEET.md              # datasheet (Gebru et al.)
├── CITATION.cff              # metadados de citação
├── LICENSE                   # licença do conjunto CC-BY-4
├── data.yaml                 # configuração YOLO (classes + caminhos)
├── images/
│   ├── train/  val/  test/   # imagens .jpg
├── labels/
│   ├── train/  val/  test/   # rótulos YOLO .txt (uma linha por objeto)
├── attributes/               # atributos de evento por imagem (.json)
├── annotation_guidelines/    # diretrizes de anotação e exemplos
└── provenance/               # proveniência e licença por amostra (.csv)
```

## Formato de anotação

Cada imagem `images/<split>/<id>.jpg` possui:

1. **Rótulo espacial (YOLO):** `labels/<split>/<id>.txt`, uma linha por objeto:
   `<class_id> <x_center> <y_center> <width> <height>` (coordenadas normalizadas [0,1]).
2. **Atributos de evento:** `attributes/<id>.json`, segundo os quatro pilares da política:
   ```
   json
   {
     "image_id": "<id>",
     "incident_class": "crime",
     "pillars": {
       "object": ["person"],
       "action": ["assault"],
       "environment": ["restricted_area"],
       "normal_abnormal": "abnormal"
     },
     "attributes": { "weapon_present": false, "theft": false },
     "source": "<fonte>", "license": "<licença_origem>"
   }
   ```

O mapeamento `class_id → nome` está em `data.yaml`. As diretrizes completas (definições, fronteiras e exemplos positivos/negativos por classe) estão em `annotation_guidelines/`.

## annotation_guidelines

| Arquivo | Conteúdo |
|---|---|
| [`class_guidelines.md`](annotation_guidelines/class_guidelines.md) | **Diretrizes por classe** (imagem): definições, fronteiras, exemplos positivos/negativos, os 4 pilares, tabela de confusões e o controle de qualidade (kappa + IoU). |
| [`video_annotation_policy.md`](annotation_guidelines/video_annotation_policy.md) | Política de anotação da **extensão em vídeo** (análise semântico-temporal; três camadas). |
| [`cvat_video_guide.md`](annotation_guidelines/cvat_video_guide.md) | Guia operacional do **CVAT** para a extensão em vídeo. |

Para o conjunto principal (imagem, v1.0), a referência é o `class_guidelines.md`.
Os dois últimos documentos cobrem o *roadmap* de vídeo (v2.0).

## Uso rápido

```python
# Treino com Ultralytics YOLO (exemplo)
from ultralytics import YOLO
model = YOLO("yolo26n.pt")           # ou yolov8n.pt / yolo11n.pt
model.train(data="data.yaml", imgsz=640, epochs=100, batch=16, seed=0)
metrics = model.val(split="test")    # avaliação no conjunto de teste
```

## Proveniência e licenças

O conjunto combina **frames de origem própria** com amostras de repositórios públicos cujas licenças **permitem redistribuição/derivação**. A proveniência é registrada por amostra em `provenance/` (fonte, licença de origem, data).

A licença do conjunto resultante é CC BY 4.0, respeitadas as licenças das fontes mantidas. Ver [LICENSE](LICENSE) e o [DATASHEET.md](DATASHEET.md).

## Ética e privacidade (LGPD)

As imagens retratam pessoas em espaços públicos. Antes da publicação, aplica-se **anonimização** (desfoque de faces e placas) e não se retêm identificadores pessoais. O conjunto destina-se a **pesquisa**; não deve ser usado para identificar indivíduos nem para decisões automatizadas de alto risco sem supervisão humana e revisão ético-legal. Ver as seções *Uses* e *Collection Process* do [DATASHEET.md](DATASHEET.md).

## Versionamento e roadmap

- **v1.0.0** — versão de imagens (detecção ao nível de imagem).
- **Roadmap:** extensão para **vídeo** com anotação temporal em três camadas (espacial/rastreamento/evento) e avaliação por tIoU; reposição completa das fontes de licença restritiva.

Mudanças são registradas em  [CHANGELOG.md](./CHANGELOG.md) e seguem versionamento semântico.

## Citação

Se utilizar este conjunto, cite o data paper (ver [CITATION.cff](CITATION.cff)):

> Pena, S. B. N.; Souza, J. R.; Nomura, S. (2026). *CCTV-SPCrime: um conjunto de dados e uma política de anotação orientada a eventos para a detecção de incidentes de segurança pública em imagens de vigilância.* `[A PREENCHER: veículo e DOI]`

## Como contribuir

Correções de anotação e relatos de problemas são bem-vindos via *issues* e *pull requests*. Ver as diretrizes em `annotation_guidelines/`.

## Contato

> Salomão Pena | Universidade Federal de Uberlândia — FACOM/UFU.
> Salomão Pena | Instituto Superior de Ciências da Educação da Huíla

---

*Recomendações:* (1) depositar uma versão arquivada em repositório com **DOI** (Zenodo, Mendeley Data ou Figshare) para findability; (2) considerar uma versão **em inglês** deste README e do datasheet para alcance internacional; (3) adicionar metadados **Croissant** (ML Commons) para interoperabilidade legível por máquina.
