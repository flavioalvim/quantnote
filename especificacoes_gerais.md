
# Especificação Completa do Sistema Quantitativo para Probabilidades de Retorno Condicionadas por Regime

Este documento consolida, em forma didática e organizada, as ideias que discutimos sobre como construir um sistema em Python para:

- Estimar probabilidades de um ativo atingir certo retorno em um horizonte de tempo (ex.: 3 dias);
- Ajustar essas probabilidades ao regime de mercado (tendência/volatilidade);
- Otimizar janelas, parâmetros e regimes usando técnicas como K-Means e algoritmo genético.

---

## 1. Motivação e problema a resolver

Queremos responder perguntas do tipo:

> Qual a probabilidade de o ativo atingir (ou superar) um certo retorno em H dias (por exemplo, 3 dias), levando em conta o regime de mercado (tendência/volatilidade) e não apenas a distribuição histórica inteira?

O método ingênuo seria:

1. Calcular todos os retornos de H dias (por exemplo, 3 dias);
2. Fazer um histograma;
3. Contar quantas vezes o retorno foi maior que um alvo X (ex.: 5%).

Exemplo: em 1000 janelas de 3 dias, 10 tiveram retorno maior que 5%. Probabilidade bruta ≈ 1%.

Problema: isso assume que:

- Todos os períodos têm a mesma “cara” (mesmo regime);
- Não há diferença entre bull market, bear market, lateralização etc.

Na prática:

- Retornos em tendência de alta têm distribuição diferente dos retornos em tendência de baixa;
- Volatilidade muda no tempo;
- Misturar tudo num balaio reduz a capacidade preditiva.

Solução:

- Separar os dados por **regimes de mercado** (bull/bear/flat, vol alta/baixa etc.);
- Calcular **probabilidades condicionais** a cada regime.

---

## 2. Conceitos fundamentais

### 2.1 Série de preços e retornos

Dados preços diários P_t:

- Retorno diário: r_t = P_t / P_{t-1} - 1
- Retorno futuro de H dias: R_{t,H} = P_{t+H} / P_t - 1
- Log-preço: L_t = log(P_t)

Trabalhar com log-preço ajuda a evidenciar tendência, pois o log-preço é a soma dos retornos.

### 2.2 Probabilidade bruta de atingir o alvo

Se você tem uma série de retornos futuros R_{t,H} e um alvo X (por exemplo, 5%), a probabilidade bruta é:

P_bruta = número de vezes em que R_{t,H} > X dividido pelo número total de observações.

Essa probabilidade ignora completamente o contexto de regime.

### 2.3 Regimes de mercado

Regimes representam “estados” qualitativos do mercado, por exemplo:

- Tendência de alta (bull);
- Tendência de baixa (bear);
- Neutro (flat);
- Volatilidade alta;
- Volatilidade baixa.

A ideia é condicionar:

P(R_{t,H} > X | Regime_t = k)

ou seja, a probabilidade do alvo depender do regime em que estamos hoje.

---

## 3. Detectando tendência local: slope da regressão linear

### 3.1 O que é o slope?

Para uma janela de tamanho W (por exemplo, 10, 15, 20 dias), olhamos os log-preços nessa janela e ajustamos uma regressão linear simples:

L = a + b * t + erro

onde:

- L é o log-preço;
- t é o índice dentro da janela (0, 1, ..., W-1);
- b é o slope (inclinação).

Interpretação:

- b > 0 → tendência de alta na janela;
- b < 0 → tendência de baixa;
- b próximo de 0 → janela neutra.

### 3.2 Por que usar log-preço e não retorno diário?

- Retornos diários são muito ruidosos, quase ruído branco.
- Regressão de retorno vs tempo tende a dar slope ≈ 0.
- O log-preço acumula os retornos e torna a tendência muito mais clara.

Por isso o método mais estável é:

- usar log-preço em janelas deslizantes;
- extrair o slope como medida de tendência local.

### 3.3 Rolling slope

Definimos uma função que, a cada dia t, calcula o slope na janela [t-W+1, ..., t]. Isso gera uma série temporal:

slope_W(t)

que indica a tendência local em cada ponto.

---

## 4. Volatilidade local

Além da tendência, é importante medir a volatilidade na mesma janela:

vol_W(t) = desvio padrão dos retornos diários na janela [t-W+1, ..., t].

Isso permite distinguir:

- bull calmo vs bull volátil;
- bear calmo vs bear em pânico;
- períodos neutros com baixa ou alta amplitude.

---

## 5. Regimes manuais (por limiares)

Uma forma simples de criar regimes é:

- Se slope_W(t) > +alpha → regime “bull”;
- Se slope_W(t) < -alpha → regime “bear”;
- Caso contrário → regime “flat”.

Opcionalmente, podemos incorporar volatilidade:

- Se vol_W(t) > beta → vol alta;
- Se vol_W(t) ≤ beta → vol baixa.

Combinações típicas:

- bull + vol baixa;
- bull + vol alta;
- bear + vol baixa;
- bear + vol alta;
- flat.

Limitação: escolher alpha e beta na mão é difícil e subjetivo.

---

## 6. Regimes automáticos com K-Means

Em vez de escolher limiares manualmente, podemos usar clusterização (K-Means) para descobrir regimes automaticamente.

### 6.1 Features para o K-Means

Para cada dia t, construímos um vetor de características (features):

- slope_W(t): tendência local;
- vol_W(t): volatilidade local;
- ret_W(t): retorno acumulado nos últimos W dias;
- opcional: volume normalizado, skewness, etc.

Assim, cada dia é um ponto em um espaço de dimensão d.

### 6.2 Pipeline do K-Means

1. Construir a matriz de features X_t baseando-se em slope, vol, etc.;
2. Padronizar as features (StandardScaler);
3. Escolher um número K de clusters (por exemplo, 3);
4. Rodar K-Means;
5. Obter labels de cluster C_t para cada dia;
6. Analisar estatísticas por cluster:
   - slope médio;
   - volatilidade média;
   - retorno futuro médio R_{t,H};
7. A partir dessas estatísticas, rotular os clusters:
   - cluster com slope médio alto e retorno futuro positivo → bull;
   - cluster com slope médio negativo e retorno futuro negativo → bear;
   - cluster com slope médio perto de 0 → flat.

---

## 7. Probabilidades condicionais por regime

Depois de rotular os regimes (manual ou via K-Means), podemos calcular:

P(R_{t,H} > X | Regime_t = k)

Para cada regime k, contamos somente os dias t que pertencem àquele regime e olhamos a frequência de R_{t,H} > X.

Resultado:

- Uma probabilidade incondicional P_bruta;
- Várias probabilidades condicionais P_k por regime.

A diferença entre essas probabilidades mostra o quanto o regime é informativo.

---

## 8. Qual janela W é “melhor”? (10, 15, 20, 30…)

A escolha da janela de tendência (W) é crítica:

- Janelas pequenas (W baixo):
  - mais sensíveis, captam microtendências;
  - mais ruidosas, trocam de regime com frequência.
- Janelas grandes (W alto):
  - mais estáveis, menos trocas;
  - mais atrasadas, demoram para reagir.

Queremos um compromisso: janelas que produzam regimes:

- bem separados em termos de comportamento futuro;
- mas estáveis o suficiente no tempo.

### 8.1 Métricas para avaliar a qualidade do regime

1. **Separação das probabilidades condicionais (ΔP)**

   Delta_P = max_k P_k - min_k P_k

   Quanto maior a diferença entre a probabilidade de atingir o alvo em regimes distintos, mais informativo o regime.

2. **Diferença de médias de retorno futuro**

   |E[R_{t,H} | bull] - E[R_{t,H} | bear]|

3. **Information Coefficient (IC)**

   Correlação entre uma pontuação (por exemplo, o próprio slope) e o retorno futuro R_{t,H}.

4. **Mutual Information**

   Mede quanto o regime (cluster) “explica” do retorno futuro.

5. **Estabilidade do regime**

   - Número de trocas de regime por ano;
   - Comprimento médio dos blocos de tempo em cada regime.

Uma janela “melhor” é aquela que produz regimes com:

- maior separação de probabilidades/médias;
- boa estabilidade (nem regime louco, nem regime morto).

---

## 9. Algoritmo Genético (GA) como otimizador

O GA não cria regimes; ele é um otimizador de parâmetros.

Ele pode escolher:

- Janela de slope (W_slope);
- Janela de volatilidade (W_vol);
- Horizonte futuro H;
- Alvo de retorno X;
- Número de clusters K;
- Quais features entram no K-Means;
- Limiar mínimo de tamanho de cluster;
- Penalizações por instabilidade de regime.

### 9.1 Cromossomo

Um cromossomo pode ser um dicionário/tabela com:

- window_slope (int);
- window_vol (int);
- horizon_H (int);
- target_X (float);
- n_clusters (int);
- usar_vol (bool);
- usar_ret_window (bool);
- etc.

### 9.2 Função de fitness

A função de fitness pode combinar:

- separação de probabilidades (ΔP);
- diferença de médias de retorno;
- IC;
- penalidade por instabilidade (número de trocas de regime).

Exemplo:

fitness = Delta_P - lambda * (num_trocas / num_dias)

O GA então evolui a população de cromossomos para maximizar o fitness e encontrar o conjunto ótimo de parâmetros.

---

## 10. Arquitetura sugerida do sistema em Python

Uma possível organização de pastas:

quant_regime_prob/
  core/
    data_loader.py
    features.py
    regimes_manual.py
    regimes_kmeans.py
    probabilities.py
    metrics.py
    ga_optimizer.py
  notebooks/
  main.py

### 10.1 Módulos principais

- data_loader.py:
  - Leitura de CSVs, limpeza, filtros.

- features.py:
  - Cálculo de log-preço, retornos diários, retornos futuros, slope, vol, retorno em janela.

- regimes_manual.py:
  - Classificação bull/bear/flat por limiar de slope (e opcionalmente vol).

- regimes_kmeans.py:
  - Construção da matriz de features;
  - Padronização;
  - K-Means;
  - Interpretação de clusters.

- probabilities.py:
  - Probabilidade bruta P_bruta;
  - Probabilidades condicionais P_k por regime.

- metrics.py:
  - Cálculo de Delta_P, IC, mutual information (se implementado), estatísticas de estabilidade.

- ga_optimizer.py:
  - Implementação do algoritmo genético;
  - Espaço de busca;
  - Função de fitness;
  - Evolução da população.

- main.py:
  - Script orquestrador:
    - Carrega dados;
    - Roda pipeline sem GA para teste;
    - Opcionalmente roda GA;
    - Exibe resultados.

---

## 11. Pipeline de uso (versão manual, sem GA)

1. Carregar a série de preços (por exemplo, de BOVA11).
2. Definir parâmetros iniciais:
   - janela W (por exemplo, 20);
   - horizonte futuro H (por exemplo, 3 dias);
   - alvo X (por exemplo, 5%).
3. Calcular:
   - retornos diários;
   - log-preço;
   - slope_W(t);
   - vol_W(t);
   - retorno acumulado na janela;
   - retorno futuro R_{t,H}.
4. Montar um DataFrame alinhado com todas essas colunas.
5. Construir a matriz de features com slope, vol e retorno acumulado.
6. Rodar K-Means e obter clusters.
7. Interpretar_clusters (assignar bull/bear/flat).
8. Calcular:
   - probabilidade bruta P_bruta;
   - probabilidades condicionais P_k;
   - métricas de separação e estabilidade.
9. Visualizar:
   - gráfico de preços com cor/área marcando o regime;
   - histogramas de R_{t,H} por regime;
   - evolução do slope e mudança de regimes.

---

## 12. Pipeline com Algoritmo Genético

1. Carregar a série de preços.
2. Definir ranges para:
   - W_slope (ex.: 5 a 60);
   - W_vol;
   - H (ex.: 1 a 10 dias);
   - X (ex.: 0.02 a 0.10);
   - K (ex.: 2 a 5).
3. Definir a função de fitness (por exemplo, Delta_P - penalidade por troca de regime).
4. Inicializar população de cromossomos (parâmetros aleatórios dentro dos ranges).
5. Para cada cromossomo:
   - Calcular features;
   - Rodar K-Means;
   - Rotular regimes;
   - Calcular probabilidades condicionais;
   - Calcular métricas;
   - Atribuir fitness.
6. Executar:
   - seleção;
   - cruzamento;
   - mutação;
   - elitismo.
7. Após N gerações, pegar o cromossomo de maior fitness.
8. Rodar o pipeline com esse cromossomo e salvar:
   - probabilidades condicionais finais;
   - interpretação dos regimes;
   - gráficos de exemplo.

---

## 13. Conclusão

Este documento descreve:

- O problema de estimar probabilidades de retorno em H dias;
- A limitação de usar apenas histogramas globais;
- A solução usando regimes de mercado;
- Como detectar tendência local via slope da regressão do log-preço;
- Como incorporar volatilidade e retorno em janela;
- Como usar K-Means para descobrir regimes automaticamente;
- Como calcular probabilidades condicionais por regime;
- Como escolher a melhor janela de tendência e outros parâmetros usando algoritmo genético;
- Uma arquitetura concreta em Python para implementar tudo.

Com isso, você tem uma especificação completa para construir, aos poucos, um sistema quantitativo flexível e poderoso para estimar probabilidades de retorno ajustadas ao regime de mercado.