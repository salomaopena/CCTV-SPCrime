# Datasheet — CCTV-SPCrime

Este documento segue o padrão **"Datasheets for Datasets"** (Gebru et al., 2021), a referência internacional para a documentação de conjuntos de dados de aprendizado de máquina.

---

## 1. Motivação

**Para que finalidade o conjunto de dados foi criado?**
Para apoiar a **avaliação reprodutível** de modelos de detecção de incidentes de segurança pública em imagens de vigilância, suprindo a escassez de conjuntos anotados que representem o **incidente como entidade semântica**. Serve de fundação empírica para a arquitetura AIVIS.GCUB (Pena et al., 2026).

**Quem criou o conjunto e em nome de qual entidade?**
Salomão Bento Nilo Pena, Jefferson R. Souza e Shigueo Nomura — Faculdade de Computação (FACOM), Universidade Federal de Uberlândia (UFU), Brasil.

**Quem financiou a criação?**
Este projeto teve o apoio de diferentes entidades:

- Coordenação de Aperfeiçoamento de Pessoal de Nível Superior (CAPES)
- Faculdade de Computação (FACOM) da Universidade Federal de Uberlândia (UFU)
- Instituto Superior de Ciências da Educação da Huíla (ISCED-Huíla)

**Comentários adicionais.**
O conjunto integra a pesquisa de doutorado do primeiro autor.

## 2. Composição

**O que representam as instâncias?**
Cada instância é uma **imagem estática** (frame) de cenário de videovigilância, com uma ou mais regiões de interesse anotadas e atributos de evento associados.

**Quantas instâncias existem no total?**
O conjunto de dados contém um conjunto de 4.042 imagens na sua primeira versão, distribuídas em **8 classes**.

**O conjunto contém todas as instâncias possíveis ou é uma amostra?**
É uma **amostra** curada de cenários de vigilância, selecionada por relevância para a segurança pública e identificabilidade visual.

**Que dado compõe cada instância?**
Imagem JPG (640×640) + rótulo espacial no formato YOLO (classe + bounding box normalizado) + atributos de evento (objeto, ação, ambiente, relação normal/anormal; e atributos específicos, p.ex., presença de arma).

**Há um rótulo ou alvo associado?**
Sim: a **classe de incidente** e os **atributos de evento**.

**Falta alguma informação em alguma instância?**
Sim: **existem imagens algumas imagens** sem todos os atributos preenchidos.

**Há relações explícitas entre instâncias?**
Não no nível de imagem. A extensão de vídeo (roadmap) introduzirá relações temporais (mesmo evento ao longo de frames).

**Há divisões recomendadas (treino/validação/teste)?**
Sim: **80% / 10% / 10%**, estratificadas por classe.

**Há erros, ruído ou redundâncias?**
Cenas ambíguas (em especial *suspicious_behavior* vs. *crime*) são intrinsecamente difíceis; o controle de qualidade da anotação é descrito na Seção 4.

**O conjunto é autocontido ou depende de recursos externos?**
**Autocontido** (imagens incluídas). A proveniência das fontes públicas é registrada para rastreabilidade.

**Contém dados confidenciais ou protegidos?**
Contém imagens de pessoas em **espaços públicos**. Aplica-se anonimização (Seção 3); não se retêm identificadores pessoais.

**Contém conteúdo potencialmente sensível ou perturbador?**
Sim — cenas de violência, acidentes e incêndio, inerentes ao domínio. Uso restrito a pesquisa (Seção 5).

## 3. Processo de coleta

**Como os dados foram adquiridos?**
Por **extração de frames** de vídeos de cenários de monitoramento de espaços públicos (origem própria) e por **integração de amostras** de repositórios públicos com licença compatível.

**Quais mecanismos/ferramentas foram usados?**
Pipeline em **Python/OpenCV** para extração de frames, filtragem de qualidade e padronização. Consultar o código no arquivo `scripts\extract_frames.py`

**Qual a estratégia de amostragem?**
Seleção orientada por classe-alvo, buscando equilíbrio entre classes e variabilidade de condições (iluminação, ângulo, oclusão).

**Em que intervalo de tempo os dados foram coletados?**
Os dados foram coleta entre os meses de novembro de 2025 a maio de 2026.

**Foram conduzidas revisões éticas?**
A coleta envolve imagens de espaço público com **anonimização** (desfoque de faces/placas) em conformidade com a **LGPD**.

**Os dados foram obtidos de terceiros ou diretamente?**
Combinação de **origem própria** e **repositórios públicos** (UCF-Crime, SCVD, Fall Vision).

## 4. Pré-processamento, limpeza e rotulagem

**Houve pré-processamento/limpeza?**
Sim: extração de frames, **filtragem de qualidade** (remoção de quadros borrados ou mal iluminados), redimensionamento para 640×640 e organização por classe.

**Os dados brutos foram preservados?**
Não. Por uma questão de privacidade e preservação da identidade, os dados brutos foram descartados e não podem ser divulgados sob qualquer circunstância.

**O software de pré-processamento está disponível?**
Sim. Todo processo de extração está disponível no repositório do github no link: `https://..`.

**Como foi feita a rotulagem?**
Anotação na ferramenta **CVAT v2.54.0**, segundo a **política orientada a eventos** (quatro pilares: objeto, ação, ambiente, relação normal/anormal). Protocolo:

- **Nº de anotadores:** 3;
- **Diretrizes:** definições, fronteiras e exemplos por classe (em `annotation_guidelines/`);
- **Resolução de divergências:** critério consenso`;
- **Concordância inter-anotadores:** `[A PREENCHER: kappa de Cohen/Fleiss sobre subconjunto de N imagens]`.

## 5. Usos

**Para quais tarefas o conjunto pode ser usado?**
Detecção de incidentes (nível de objeto/evento) em imagens; **benchmark** de detectores; estudo de **metodologia de anotação** orientada a eventos.

**Há algo no conjunto que possa afetar usos futuros?**
A heterogeneidade de fontes e a ambiguidade de classes contextuais podem introduzir **viés**; imagens estáticas **não** capturam a dimensão temporal de certos incidentes.

**Há tarefas para as quais o conjunto NÃO deve ser usado?**
**Não** deve ser usado para: identificar indivíduos; vigilância em produção sem revisão ético-legal; decisões automatizadas de alto risco sem supervisão humana; qualquer uso que viole a LGPD ou as licenças das fontes.

**Comentários sobre mitigação de riscos.**
Recomenda-se supervisão humana, avaliação de viés por classe e cenário, e conformidade legal antes de qualquer aplicação operacional.

## 6. Distribuição

**Como o conjunto será distribuído?**
Via **GitHub** e, recomendadamente, um repositório arquivado com **DOI** (Zenodo, Mendeley Data ou Figshare). `[A PREENCHER: URL e DOI]`

**Sob qual licença?**
`[A PREENCHER: CC BY 4.0 ou CC0]`, respeitadas as licenças das fontes públicas mantidas.

**Há restrições de uso ou de exportação?**
Nenhuma além da licença escolhida e das licenças de origem registradas na proveniência; fontes com cláusula de não-derivação foram excluídas.

**Quando será distribuído?**
`[A PREENCHER: data de publicação da versão 1.0.0]`.

## 7. Manutenção

**Quem mantém o conjunto?**
`[A PREENCHER: nome e e-mail institucional do mantenedor]` — FACOM/UFU.

**Como reportar erros ou contribuir?**
Por *issues* e *pull requests* no repositório.

**O conjunto será atualizado?**
Sim. Roadmap: extensão para **vídeo** com anotação temporal (três camadas). Atualizações seguem **versionamento semântico** e são registradas no CHANGELOG.

**Versões antigas serão mantidas?**
`[A PREENCHER: política de retenção de versões]`.

**Há mecanismo para contribuições externas?**
Sim, via *pull requests*, sujeitas à revisão de qualidade segundo as diretrizes de anotação.

---

### Referência do padrão

Gebru, T. et al. (2021). *Datasheets for Datasets.* Communications of the ACM, 64(12), 86–92.
