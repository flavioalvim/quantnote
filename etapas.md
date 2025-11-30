1- Instalar ambeiente virtual em .venv
2- Criar requirements.txt
3- Criar classe que acessa yahoo e retorna dataframe pandas ohlcv ja com precos ajustados (o maior periodo)
4- Criar interface de repositore
5- Criar uma pasta data
6- criar uma classe que recebe um dataframe e grava em data em parquet com nome com sufixo que inclui o tiker e a data e hora da ultima atualizacao
7 - Criar uma classe que le esse artquivo parque e retorna o dataframe
8- Criar uma classe ou funcao utilitaria que calcula o log retorno de dois precos consecutivos e outra para transformar log_retorno e retorno percentual e vice versa
9- criar um arquivo de configuração com as seguintes metricas:

-retorno_futuro (em periodos) #O log retorno em X dias 

-window_log_retorno_movel #tamanho da janela onde calcularemos aas somas dos log_retornos para ter o log_retorno do periodo que corresponde ao tamnho da janela passada. 

-window_slope (int); # tamanho da janela para calculo do slope

-window_vol (int);#tamanho da janela para calculo da volailidade

-n_clusters (int) - #numero de clustar para k means

10 - criar classe interface com metodo add_column onde criaremos classes cncretas para adicionar ao dataframe ohlcv, cada coluna com indicadoes e valores distintos

11- criar classe concreta para adicionar as colunas:
-log_preco_fechamento
-log_retorno_periodo (passado)
-log_retorno_futuro_x_periodos (janela com (retorno_futuro_em_periodos) periodos a diante (nao é passado , é futuro mesmo (é o que queremos prever)))
-volatilidade_movel_de_x_periodos - usar window_vol
-slope_movel - uasr window_slope

- criar classe para receber esse dataframe novo com as colunas aicionadas e plotar um histogram com os log_retornos_futuros_x_dias geral e outra para o histograma condicional para 6 grupos:
-tendencia_alta_com_vol_alta
-tendencia_alta_com_vol_baixa
-tendencia_baixa_com_vol_alta
-tendencia_baixa_com_vol_baixa
-lateral_com_vol_alta
-latera_com_vol_baixa

Criar um notebook onde cada celula e uma classe e ao final das definicoes de classes criaremos o passo a passo para rodar o sistema

