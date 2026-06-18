# CCTV-SPCrime: um conjunto de dados e uma política de anotação orientada a eventos para a detecção de incidentes de segurança pública em imagens de vigilância

**Salomão Bento Nilo Pena · Jefferson R. Souza · Shigueo Nomura**
Faculty of Computer Science (FACOM), Universidade Federal de Uberlândia — MG, Brasil
{salomao.pena, jrsouza, shigueonomura}@ufu.br

> *Nota de formatação:* as citações estão em estilo autor-ano entre parênteses para facilitar a edição; converter ao estilo numérico do veículo no template final. Marcadores `[A PREENCHER: …]` indicam valores experimentais a inserir antes da submissão. A seção final "Pendências" deve ser removida do PDF de submissão.

---

## Resumo

Sistemas de videovigilância urbana produzem volumes de dados visuais cuja análise manual é inviável em larga escala. Grande parte das soluções de detecção automática é centrada em objetos isolados e treinada sobre conjuntos de dados anotados ao nível de objeto ou com supervisão fraca ao nível de vídeo, o que limita a representação de incidentes como entidades semânticas e dificulta a sua reprodutibilidade. Este artigo apresenta o **CCTV-SPCrime**, um conjunto de dados de imagens de vigilância anotado segundo uma **política orientada a eventos**, abrangendo oito classes de incidentes de segurança pública (acidente, comportamento suspeito, crime, incêndio, intrusão, objeto suspeito, queda e vandalismo). A contribuição é tripla: (i) o conjunto de dados, com proveniência e licença documentadas; (ii) a política de anotação orientada a eventos, com protocolo e controle de qualidade, incluindo concordância inter-anotadores; e (iii) um **benchmark de detecção** que compara variantes da família YOLO (YOLOv8n, YOLO11n e YOLO26n) sob a mesma divisão de dados, com configuração de treino totalmente reportada. No melhor detector avaliado, obtém-se mAP@50 médio de **0,93** no conjunto de teste, com desempenho superior nas classes visualmente mais discriminativas. O escopo deste trabalho é a detecção ao nível de imagem; a arquitetura multiagente completa, com raciocínio temporal e explicabilidade, é apresentada em trabalho companheiro (Pena et al., 2026). O conjunto de dados destina-se a apoiar a avaliação reprodutível de modelos de detecção de incidentes em vigilância.

**Palavras-chave:** conjunto de dados; anotação orientada a eventos; detecção de incidentes; vigilância por vídeo; segurança pública; YOLO.

## I. Introdução

O uso de videomonitoramento em ambientes residenciais, comerciais e públicos consolidou-se como instrumento de apoio à segurança pública e à gestão urbana. As câmeras geram grandes volumes de dados visuais cuja análise manual é inviável em cenários reais, o que motivou a automação da análise de vídeo como requisito de escalabilidade (Shidik et al., 2019; Cabanillas-Carbonell et al., 2025).

A maioria das soluções de detecção em tempo real, em especial as baseadas na família YOLO, permanece **centrada em objetos**: identifica entidades isoladas — pessoas, veículos, armas, objetos abandonados — e infere o incidente por regras heurísticas ou pós-processamento (Hanavi e Hidayat, 2020; Omarov et al., 2022; Lenka et al., 2025). Esse paradigma limita a representação de incidentes que dependem de interação e contexto. Em paralelo, a detecção de anomalias define a anomalia de forma estatística e frequentemente opera sob classificação binária (normal/anormal) ou rótulos de ação isolados (Duong et al., 2023; Kumar et al., 2025).

Um obstáculo central, anterior à arquitetura de qualquer sistema, é a **escassez de conjuntos de dados anotados de forma a representar o incidente como unidade semântica**. Muitos conjuntos públicos são coletados em ambientes controlados, baseiam-se em clipes de filmes ou esportes, ou adotam anotação ao nível de vídeo de baixa resolução temporal, o que compromete a generalização e a reprodutibilidade (Sultani et al., 2018; Acsintoae et al., 2022). É essa lacuna de dados e de metodologia de anotação que este trabalho aborda.

Este artigo concentra-se, portanto, na **fundação de dados** para a detecção de incidentes de segurança pública em imagens de vigilância. As contribuições são:

- **O conjunto de dados CCTV-SPCrime**, com imagens de cenários de vigilância anotadas para oito classes de incidentes, com proveniência e licença documentadas;
- **Uma política de anotação orientada a eventos**, que trata o incidente como entidade semântica composta (objeto, ação, ambiente e relação normal/anormal), com protocolo e controle de qualidade (concordância inter-anotadores);
- **Um benchmark de detecção** sobre o CCTV-SPCrime, comparando variantes YOLO sob a mesma divisão e com configuração de treino reportada na íntegra, fornecendo uma linha de base reprodutível.

A arquitetura multiagente que consome este conjunto de dados — incluindo raciocínio temporal e explicabilidade — não é objeto deste artigo e é apresentada em trabalho companheiro (Pena et al., 2026); o presente trabalho restringe-se à detecção ao nível de imagem e à disponibilização do recurso de dados. A Seção II discute trabalhos relacionados, com ênfase em conjuntos de dados e estratégias de anotação; a Seção III descreve a construção do CCTV-SPCrime; a Seção IV detalha a política de anotação; a Seção V apresenta o benchmark de detecção; e a Seção VI conclui.

## II. Trabalhos Relacionados

### A. Análise inteligente de vídeo e detecção por objetos

A evolução do videomonitoramento para a análise inteligente de vídeo é documentada em revisões sistemáticas recentes (Shidik et al., 2019; Cabanillas-Carbonell et al., 2025), que apontam o predomínio de arquiteturas hierárquicas e de detectores de estágio único. A família YOLO consolidou-se como padrão para detecção em tempo real em segurança física, dada a relação favorável entre velocidade e precisão (Ali e Zhang, 2024; Chatterjee et al., 2024). Tais abordagens, contudo, concentram-se na presença de entidades isoladas e requerem mecanismos adicionais para inferir o significado semântico de incidentes complexos.

### B. Detecção de anomalias e de comportamento

Trabalhos de detecção de anomalias combinam representações espaciais e temporais para identificar desvios (Dwivedi et al., 2023; Ganagavalli e Santhi, 2024; Lenka et al., 2025). Apesar do bom desempenho em benchmarks, muitos operam sob supervisão fraca ao nível de vídeo ou rótulos de ação isolados, o que dilui a semântica do incidente (Omarov et al., 2022; Kumar et al., 2025). Note-se que a integração de modelagem semântica, raciocínio temporal e explicabilidade numa única arquitetura é tratada no trabalho companheiro (Pena et al., 2026); aqui interessa o substrato de dados que torna tal avaliação possível e reprodutível.

### C. Conjuntos de dados e estratégias de anotação

O desempenho de modelos de aprendizado profundo depende da qualidade e da representatividade dos dados. Conjuntos públicos como UCF-Crime (Sultani et al., 2018), UBnormal (Acsintoae et al., 2022), SCVD (Aremu et al., 2024) e Fall Vision (Rahman et al., 2024; 2025) tornaram-se referências, mas apresentam limitações para a vigilância de segurança pública: cenários restritos (esportes, multidões específicas), anotação centrada no objeto ou rótulo de vídeo, e baixa variabilidade de condições reais (iluminação, clima, oclusão).

Uma distinção metodológica relevante é a granularidade da anotação. A anotação **ao nível de vídeo** atribui um rótulo global ao clipe — econômica, porém de baixa resolução temporal e sujeita a ruído de supervisão fraca. A anotação **ao nível de quadro/evento** permite localização precisa, ao custo de maior esforço. A Tabela I sintetiza o contraste; é nessa lacuna — a ausência de conjuntos com anotação orientada a eventos para incidentes de segurança pública — que o CCTV-SPCrime se posiciona.

**Tabela I — Anotação ao nível de vídeo vs. ao nível de quadro/evento**

| Atributo | Nível de vídeo (supervisão fraca) | Nível de quadro/evento (supervisão forte) |
|---|---|---|
| Escopo temporal | Rótulo global do clipe | Localização precisa de quadro/instância |
| Esforço | Reduzido (passagem única) | Elevado (anotação por quadro/evento) |
| Precisão semântica | Baixa para eventos breves | Alta; permite localização e rastreamento |
| Reprodutibilidade | Limitada | Maior, com protocolo e controle de qualidade |

A Tabela II posiciona o CCTV-SPCrime frente a conjuntos públicos, segundo dimensões comparáveis (tarefa, tipo de anotação, classes e limitação) — e não segundo métricas heterogêneas de modelos.

**Tabela II — Posicionamento do CCTV-SPCrime frente a conjuntos públicos**

| Conjunto | Domínio/Tarefa | Tipo de anotação | Nº de classes | Limitação principal |
|---|---|---|---|---|
| UCF-Crime (Sultani et al., 2018) | Anomalia em vídeo | Nível de vídeo (fraca) | 13 | Localização temporal grosseira |
| UBnormal (Acsintoae et al., 2022) | Anomalia open-set | Quadro (sintético) | — | Domínio sintético; licença restritiva |
| SCVD (Aremu et al., 2024) | Violência armada | Vídeo/imagem | — | Foco estreito (violência armada) |
| Fall Vision (Rahman et al., 2024) | Queda | Vídeo | — | Classe única |
| **CCTV-SPCrime (este trabalho)** | **Incidentes de segurança pública (imagem)** | **Orientada a eventos (bbox + atributos)** | **8** | **Imagem estática; sem dimensão temporal (ver trabalho futuro)** |

## III. O Conjunto de Dados CCTV-SPCrime

### A. Construção

O CCTV-SPCrime foi construído a partir de imagens de cenários de monitoramento de espaços públicos — ambientes urbanos externos e internos, como ruas, praças, áreas de circulação e zonas de acesso controlado —, caracterizados por variações de iluminação, resolução, ângulo de câmera, oclusões parciais e ruído visual típicos de CCTV (Kim et al., 2019).

Para mitigar a ausência de categorias específicas, integraram-se amostras de repositórios públicos consolidados: o Fall Vision (Rahman et al., 2024; 2025) para a classe de quedas, e o SCVD (Aremu et al., 2024) e o UCF-Crime (Sultani et al., 2018) para o suporte às demais classes.

> **[A PREENCHER / AÇÃO OBRIGATÓRIA]** O UBnormal (Acsintoae et al., 2022) foi **removido** desta versão do conjunto: a sua licença CC BY-NC-ND proíbe a distribuição de obras derivadas (reanotação/recomposição). As amostras antes oriundas do UBnormal estão sendo repostas por dados próprios (encenados e/ou sintéticos). Recontabilizar os totais por classe após a remoção e a reposição, e atualizar a Tabela III.

O processamento foi realizado por um pipeline em Python/OpenCV para extração automatizada de quadros que exibem os comportamentos-alvo, conversão para JPG, **filtragem de qualidade** (remoção de quadros borrados ou com iluminação insuficiente), redimensionamento para 640×640 px e organização por classe, com geração de rótulos no formato YOLO (Fig. 1 — *workflow*).

### B. Definição das classes de incidentes

As classes foram definidas pela relevância para a segurança pública e pela identificabilidade visual em imagens estáticas, estruturadas para representar **eventos** e não objetos isolados:

- **Acidente:** colisão de veículos, quedas em massa, emergências médicas visíveis.
- **Comportamento suspeito:** movimentos atípicos, pessoa correndo fora de contexto (fuga, pânico), perseguição rápida e intencional.
- **Crime:** violência física ou tentativa de subtração de bens (agressão, socos, empurrões, conflito físico, arrombamento, furto, roubo, ameaça com arma, briga). Atributos descrevem a natureza do evento e a presença de armamento.
- **Incêndio:** fogo, fumaça visível, chamas em objetos/veículos/propriedades.
- **Intrusão:** transposição de perímetro, cercas, portões ou áreas proibidas.
- **Objeto suspeito:** mochilas, malas, pacotes abandonados por tempo anormal, armas.
- **Queda:** pessoa caída em postura anômala (acidente ou desmaio).
- **Vandalismo:** danos à propriedade (quebra de vidros, destruição).

Seguindo a consolidação proposta por Sultani et al. (2018), ocorrências agressivas e delituosas foram unificadas sob a categoria **crime**, com atributos que distinguem assalto, violência, furto e conflito — útil quando as fronteiras semânticas são pouco definidas em imagens estáticas.

### C. Estatísticas e divisão

O conjunto adota divisão de 80% treino, 10% validação e 10% teste (Tabela III). *Os totais abaixo refletem a versão anterior à remoção do UBnormal e serão atualizados (ver III-A).*

**Tabela III — Estatísticas do CCTV-SPCrime por classe** `[A PREENCHER: recontar após remoção do UBnormal]`

| Classe | Treino | Validação | Teste | Total |
|---|---|---|---|---|
| Acidente | 400 | 50 | 50 | 500 |
| Comportamento suspeito | 404 | 51 | 51 | 506 |
| Crime | 455 | 57 | 57 | 569 |
| Incêndio | 420 | 52 | 52 | 524 |
| Intrusão | 426 | 53 | 53 | 532 |
| Objeto suspeito | 400 | 50 | 50 | 500 |
| Queda | 326 | 41 | 41 | 408 |
| Vandalismo | 403 | 50 | 50 | 503 |
| **Total** | **3.234** | **404** | **404** | **4.042** |

### D. Licença, governança de dados e ética

As imagens de origem própria são anonimizadas (desfoque de faces e placas) em conformidade com a LGPD. A proveniência é registrada por amostra (fonte, licença de origem, data). O conjunto resultante é disponibilizado sob licença **CC BY 4.0** (compatível com os componentes CC BY incluídos), respeitadas as licenças das fontes públicas mantidas. Fontes com cláusula de não-derivação (p.ex., UBnormal) foram excluídas. `[A PREENCHER: link do repositório e DOI]`

## IV. Política de Anotação Orientada a Eventos

A política trata o **incidente como unidade central de análise**. Cada incidente é anotado por uma região de interesse que delimita a área do evento e engloba os elementos relevantes para a sua compreensão, organizados em quatro pilares (Fig. 2):

1. **Objeto (quem?):** pessoa, veículo, objeto suspeito;
2. **Ação (o quê?):** agredir, pular cerca, quebrar vidro, cair;
3. **Ambiente (onde?):** zona proibida, horário noturno, proximidade de multidão;
4. **Relação normal/anormal (por quê?):** p.ex., correr para o ônibus = normal; correr após furto = anormal.

Essa estrutura permite uma análise interpretativa da cena e mitiga falsos positivos decorrentes de ações benignas visualmente similares a incidentes. **Exemplo concreto** (Fig. 3 — caso anotado): em uma cena de calçada, o objeto *pessoa* executa a ação *correr*; o ambiente é *ponto de ônibus*; não há perseguidor nem objeto subtraído; logo, a relação é *normal* e a instância **não** é rotulada como comportamento suspeito — ao passo que a mesma morfologia de movimento, em *zona de acesso controlado* e após uma ação de *arrombamento*, é rotulada como evento composto.

### A. Ferramenta e protocolo

A anotação foi realizada na ferramenta **CVAT v2.54.0** (CVAT.ai/Intel), escolhida após comparação com labelImg e Roboflow, pela flexibilidade e pelo suporte a anotação semiautomática com atributos semânticos. O protocolo de anotação seguiu:

- **Anotadores:** `[A PREENCHER: número de anotadores]`, com diretrizes escritas por classe (definição, fronteiras e exemplos positivos/negativos);
- **Resolução de divergências:** `[A PREENCHER: critério — p.ex., terceiro anotador/consenso]`;
- **Concordância inter-anotadores:** `[A PREENCHER: valor de kappa (Cohen/Fleiss) sobre subconjunto de N imagens]`, calculada sobre a classe atribuída e a sobreposição de regiões (IoU);
- **Controle de qualidade:** revisão por amostragem e verificação de consistência de atributos.

A documentação completa do protocolo e das diretrizes acompanha o repositório do conjunto de dados.

## V. Benchmark de Detecção

### A. Configuração experimental

Os experimentos foram conduzidos em estação com processador Intel Core i7-12700H (14 núcleos, 2,3–4,7 GHz), 32 GB de RAM DDR4 e GPU NVIDIA RTX 3060 (6 GB GDDR6), sob Ubuntu 22.04 LTS, Python 3.11, PyTorch 2.x e OpenCV 4.12.0. Os detectores foram inicializados com pesos pré-treinados no COCO e submetidos a *fine-tuning* sobre o CCTV-SPCrime.

**Hiperparâmetros (reportados para reprodutibilidade):** imgsz = 640; batch = 16; otimizador SGD (momentum = 0,937); learning rate inicial = 0,01 (lrf = 0,01); weight decay = 0,0005; épocas = 100; early stopping (patience = 20); seed = 0. O limiar de confiança foi fixado em **0,37**, valor que **maximiza o F1 no conjunto de validação** (curva F1–confiança), e não o padrão 0,5; o efeito do limiar sobre falsos positivos/negativos é reportado na análise de erros.

A avaliação reporta **mAP@50** e mAP@50:95, além de precisão (P) e recall (R) por classe, **sobre o conjunto de teste** (404 amostras), distinto do conjunto de validação usado para seleção de limiar.

### B. Baselines sob a mesma divisão

Para situar o desempenho, comparam-se três variantes da família YOLO sob a **mesma divisão** treino/validação/teste (Tabela IV). YOLO26 é a versão mais recente (Jocher e Qiu, 2026); YOLOv8n e YOLO11n são incluídos por serem amplamente adotados.

**Tabela IV — Comparação de baselines no conjunto de teste (mesma divisão)** `[A PREENCHER: executar v8n e 11n]`

| Modelo | mAP@50 | mAP@50:95 | P (média) | R (média) |
|---|---|---|---|---|
| YOLOv8n | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` |
| YOLO11n | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` | `[A PREENCHER]` |
| YOLO26n (adotado) | 0,93 | `[A PREENCHER]` | 0,90 | 0,90 |

### C. Resultados por classe

A Tabela V apresenta o desempenho por classe do detector adotado (YOLO26n) no conjunto de teste.

**Tabela V — Resultados por classe (YOLO26n, conjunto de teste)**

| Classe | Instâncias | P | R | mAP@50 |
|---|---|---|---|---|
| Acidente | 561 | 0,89 | 0,89 | 0,92 |
| Comportamento suspeito | 589 | 0,73 | 0,79 | 0,82 |
| Crime | 525 | 0,98 | 0,96 | 0,98 |
| Incêndio | 570 | 0,99 | 0,94 | 0,98 |
| Intrusão | 582 | 0,90 | 0,94 | 0,96 |
| Objeto suspeito | 665 | 0,91 | 0,89 | 0,94 |
| Queda | 544 | 0,87 | 0,94 | 0,92 |
| Vandalismo | 570 | 0,91 | 0,84 | 0,91 |
| **Média** | — | **0,90** | **0,90** | **0,93** |

Os resultados (apresentados acima) indicam, **na sequência, a seguinte interpretação**: classes com padrões visuais mais discriminativos — incêndio, crime e intrusão — atingem os maiores valores de mAP@50, enquanto comportamento suspeito apresenta o menor desempenho (0,82), consistente com a sua ambiguidade visual já reportada na literatura (Omarov et al., 2022; Kumar et al., 2025). As curvas de perda de treino e validação convergem ao longo de 100 épocas sem sinais de sobreajuste (Fig. 4); a matriz de confusão (Fig. 5) e a curva F1–confiança (Fig. 6) corroboram o equilíbrio entre precisão e recall no limiar adotado.

### D. Análise de erros e limitações de escopo

Os principais erros concentram-se em cenas ambíguas, sobretudo a confusão entre crime e interações sociais não violentas, em que a evidência **estática** é insuficiente. Esta é uma limitação **de escopo, e assumida**: o presente trabalho avalia detecção ao nível de imagem; a discriminação de eventos que dependem de dinâmica temporal requer informação de vídeo e é tratada na arquitetura companheira (Pena et al., 2026), constituindo, aqui, trabalho futuro. Nenhuma afirmação de interpretação semântica, raciocínio temporal ou explicabilidade é feita com base nos resultados deste artigo.

## VI. Conclusão

Este artigo apresentou o **CCTV-SPCrime**, um conjunto de dados de imagens de vigilância anotado segundo uma **política orientada a eventos** para oito classes de incidentes de segurança pública, acompanhado de um **benchmark de detecção** reprodutível com variantes da família YOLO. As contribuições — o conjunto de dados com proveniência e licença documentadas, a metodologia de anotação com controle de qualidade, e a linha de base de detecção — visam suprir a escassez de recursos anotados ao nível de evento para o domínio.

O escopo é deliberadamente a detecção ao nível de imagem; a arquitetura multiagente com raciocínio temporal e explicabilidade que consome este conjunto de dados é apresentada em trabalho companheiro (Pena et al., 2026). Como trabalho futuro, destacam-se a extensão do conjunto para **vídeo** com anotação temporal, a reposição completa das fontes de licença restritiva por dados próprios e a expansão para novos cenários urbanos. O conjunto de dados e a documentação de anotação estão disponíveis em `[A PREENCHER: URL do repositório]`.

## Referências

> Reaproveitar as entradas do manuscrito original e acrescentar as novas. Lista em ordem de aparição (autor-ano):

- Shidik, G. F. et al. (2019). A systematic review of intelligence video surveillance. *IEEE Access*, 7.
- Cabanillas-Carbonell, M. et al. (2025). Artificial intelligence in video surveillance systems. *Adv. Sci. Technol. Res. J.*, 19(3).
- Hanavi; Hidayat, F. (2020). Intelligent video analytic for suspicious object detection. *ICISS*.
- Omarov, B. et al. (2022). State-of-the-art violence detection techniques. *PeerJ Computer Science*, 8.
- Lenka, G. G. et al. (2025). AlertEye: Real-time surveillance and threat identification. *AISTS*.
- Duong, H.-T. et al. (2023). Deep learning-based anomaly detection in video surveillance: a survey. *Sensors*, 23(11).
- Kumar, P. et al. (2025). Comparative analysis of suspicious activity detection techniques. *ICDT*.
- Sultani, W.; Chen, C.; Shah, M. (2018). Real-world anomaly detection in surveillance videos. *CVPR*.
- Acsintoae, A. et al. (2022). UBnormal: new benchmark for supervised open-set video anomaly detection. *CVPR*.
- Aremu, T. et al. (2024). SSIVD-Net (Smart-City CCTV Violence Detection). *Intelligent Computing*, Springer.
- Rahman, N. N. et al. (2024; 2025). Fall Vision: a benchmark video dataset for fall detection. *Harvard Dataverse; Data in Brief*, 59.
- Ali, M. L.; Zhang, Z. (2024). The YOLO framework: a comprehensive review. *Computers*, 13(12).
- Chatterjee, N. et al. (2024). YOLOv8-based intrusion detection system. *ICRITO*.
- Nasir, R. et al. (2025). Real-time dense crowd abnormal behavior detection using YOLOv8 (CADF). *Artificial Intelligence Review*, 58(7).
- Ganagavalli, K.; Santhi, V. (2024). YOLO-based anomaly activity detection. *Signal, Image and Video Processing*, 18.
- Kim, G. et al. (2019). Specific area intrusion detection using YOLO in CCTV. *IJITEE*, 8(8).
- Shanthi, P.; Manjula, V. (2025). CNN-YOLO techniques for face and weapon detection. *Discover Computing*, 28(1).
- Negre, P. et al. (2024). Literature review of deep-learning-based detection of violence in video. *Sensors*, 24(12).
- Jocher, G.; Qiu, J. (2026). Ultralytics YOLO26. github.com/ultralytics/ultralytics.
- **Pena, S. B. N.; Souza, J. R.; Nomura, S. (2026). AIVIS.GCUB: arquitetura multiagente explicável para reconhecimento semântico-temporal de incidentes em vigilância por vídeo. (trabalho companheiro).** `[A PREENCHER: veículo/estado]`
- **CVAT.ai (2024). Computer Vision Annotation Tool (CVAT), v2.54.0.** `[A PREENCHER: referência/URL]`
- Cohen, J. (1960). A coefficient of agreement for nominal scales. *Educational and Psychological Measurement*, 20(1). · Fleiss, J. L. (1971). Measuring nominal scale agreement among many raters. *Psychological Bulletin*, 76(5).

---

## Pendências para submissão (remover do PDF final)

1. **Recontar** o CCTV-SPCrime após remover o UBnormal e repor por dados próprios; atualizar a Tabela III, os totais e os contadores de instâncias.
2. **Executar os baselines** YOLOv8n e YOLO11n no mesmo split e preencher a Tabela IV; preencher mAP@50:95 do YOLO26n.
3. **Calcular e inserir o kappa** inter-anotadores e os números do protocolo (nº de anotadores, critério de divergência).
4. **Inserir a Fig. 3** com um caso real de anotação de evento (os 4 pilares sobre uma imagem anonimizada).
5. **Preencher hiperparâmetros** faltantes (otimizador, weight decay, seed) e confirmar o ambiente (PyTorch/CUDA).
6. **Definir e declarar a licença** do conjunto (CC BY 4.0 ou CC0) e o **DOI/URL** do repositório; confirmar disponibilidade pública.
7. **Confirmar a citação do trabalho companheiro** (AIVIS.GCUB) e seu estado/veículo, garantindo a separação de escopo.
8. **Converter citações** para o estilo numérico do veículo e revisar formatação (posição de figuras/tabelas, resíduos de template, título dentro do limite).
9. **Escolher o veículo** distinto do AIVIS.GCUB (não submeter ambos ao mesmo evento).
