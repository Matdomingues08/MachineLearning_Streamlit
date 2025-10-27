import pandas as pd
import joblib
import os
from sklearn import model_selection, preprocessing, pipeline, linear_model, metrics

#etapa 1 carregador dados

def carregador_dados(caminho_arquivo='historicoAcademico.csv'):
    try:
        #carregamento dos dados
        if os.path.exists(caminho_arquivo):

            df = pd.read_csv(caminho_arquivo, encoding= 'latin1', sep=',')

            print('o arquivo foi carregado com sucesso!')

            return df
        else:
            print('o arquivo não foi encontrado dentro da pasta')

            return None
    except Exception as e:
        print('erro inesperado ao carregar o arquivo: ', e)

        return None
    
dados = carregador_dados()


#--------- ETAPA 02 : PREPARAÇÃO E DIVISÃO DE DADOS
#Definicção de x (features) e y (target)

if dados is not None:
    print(f"\nTotal de regsitros carregados{len(dados)}")
    print("iniciando o pipeline de treinamento")

    TARGET_COLUMN = 'Status_Final'
    #etapa 2.1 = definição das features e target 
    try:
        x = dados.drop (TARGET_COLUMN, axis=1)
        y = dados[TARGET_COLUMN]
        print(f"Features (x) definidaas: {list(x.columns)}")
        print(f"Features (y) definidas: {TARGET_COLUMN}")

    except KeyError:

        print(f"\n----Erro Critico----")
        print(f"A coluna {TARGET_COLUMN} não foi encontrado no CSV")
        print(f"Colunas disponiveis: {list(dados.columns)}")
        print(f"Por favor, ajuste a variavel 'TARGET_COLUMN' e tente novamente")
        #se o target não for encontrado irá encerrar o script!
        exit()

    #etapa 2.2 - divisão entre treino e teste
    print("\n----- Dividindo dados e em treino e teste -----")

    X_train, X_test, y_train, y_test = model_selection.train_test_split(x, y,
        test_size=0.2,  #20%% dos dados serão utilizados para teste
        random_state=42, #garantir a reprodutibidade
        stratify=y       #Manter a proporção aprovados e reprovados
    )
    print(f"Dados de treino: {len(X_train)}) | Dados de teste: {len(X_test)}")

    # etapa 3: criação da pipeçine de ML

    print("\nCriando a pipeline da ML...-----")
    pipeline_model = pipeline.Pipeline([  
        ('scaler', preprocessing.StandardScaler()),
        ('model', linear_model.LogisticRegression(random_state=42))
    ])
    #etapa 04 
    print("\n---------TREINAEMNTO DE MODELO...-----------") 
    #treina a pipeline com os dados de treino
    pipeline_model.fit(X_train, y_train)

    print("modelo treinado. Avaliando com os dados de teste")
    Y_pred = pipeline_model.predict(X_test)

    #AVALIAÇÃO D DESEMPENHO

    accuracy = metrics.accuracy_score(y_test, Y_pred)
    report = metrics.classification_report(y_test, Y_pred)

    print("\n-----Relatorio de avaliação geral------")
    print(f"Acuráia GEral: {accuracy * 100:2f}%")
    print("\nRelátorio de classificação detalhado")
    print(report)


    #Etapa 5: salvando o modelo
    model_filename = 'modelo_previsao_desempenho.joblib'

    print(f"\nsalvando o pipeline treinado em.. {model_filename}")
    joblib.dump(pipeline_model, model_filename)


    print("Processo concluido com sucesso")

    print(" arquivo '{model_filename}' está para ser utilizado")

else:
    print ("O pipeline não pode continuar pois os dados não dorma carregados!")