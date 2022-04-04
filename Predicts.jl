function Fₘ(a, learning_rate, X, f₀)
    tamanho = size(a)[1] #pega o número de árvores do modelo
    rows = size(X)[1] #pega o número de dados do conjunto de entrada X
    f_pred = ones(rows)*f₀ #modelo inicial, f₀
    #i = 1 #contador para o predict (?)
    #R = [] #Resultado (?)
    resposta = zeros(rows,tamanho+1) #final + f0
    resposta[1:end,1] = f_pred
    resposta_arvore = zeros(rows) #resposta de todos os dados por árvore

    for arvore = 1:tamanho # para cada árvore construída
        # a seguir temos os vetores que formam uma árvore de decisão
        isnt_leave = Int64.(a[arvore][6]) # é folha?
        feature = Int64.(a[arvore][4]) #coluna do conjunto utilizada naquele nó
        threshold = a[arvore][3] #valor de comparação do ≤
        children_left = Int64.(a[arvore][1]) #nós a esquerda
        children_right = Int64.(a[arvore][2]) #nós a direita
        gammas = a[arvore][7] #predict encontrado no treino
        for k = 1:rows
            onde = 1 #posição do dado na árvore para saber em qual nó estamos
            while (isnt_leave[onde]==true) # verifica se o nó onde o dado está é folha ou nó
                if X[k,feature[onde]+1] ≤ threshold[onde] # condição do nó, if true -> foi para esquerda
                    onde = children_left[onde]+1
                else #caso false -> foi para nó direito
                    onde = children_right[onde]+1
                end
            end
            #println("----")
            #if contador%1 == 0
                #if length(resultado) !=0
                    #println("quase resposta:", sum(resultado)+f₀)
                #end
            #end
            #println(gammas[onde])
            #println("γ:",gammas[onde])

            resposta_arvore[k] = learning_rate*gammas[onde]
            #push!(resultado,gammas[onde]) # armazena os resultados das etapas
            #if k <= 3 && arvore <= 2
            #    println("----------")
            #    println("k:",k)
            #    println("arvore:",arvore)
            #    println(resposta[1:rows,tamanho])
            #    push!(R,sum(resultado)+f₀)
            #end
        end
        resposta[1:end,arvore+1] = resposta_arvore
    end

    #for k = 1:rows # para todos os dados do conjunto X
    #    resultado = [] #armazeno as respostas de todas as árvores para aquele dado

    #    f_pred[k] += sum(resultado)
    #end
    return sum(resposta,dims=2)#, r2(y_real,sum(resposta,dims=2))
end

function Fₘ_momentum(a, X, f₀)
    #println("F0",f₀)
    tamanho = size(a)[1] #pega o número de árvores do modelo
    rows = size(X)[1] #pega o número de dados do conjunto de entrada X
    f_pred = ones(rows)*f₀ #modelo inicial, f₀
    #i = 1 #contador para o predict (?)
    #R = [] #Resultado (?)
    resposta = zeros(rows,tamanho+1) #final + f0
    resposta[1:end,1] = f_pred
    resposta_arvore = zeros(rows) #resposta de todos os dados por árvore

    for arvore = 1:tamanho # para cada árvore construída
        # a seguir temos os vetores que formam uma árvore de decisão
        isnt_leave = Int64.(a[arvore][6]) # é folha?
        feature = Int64.(a[arvore][4]) #coluna do conjunto utilizada naquele nó
        threshold = a[arvore][3] #valor de comparação do ≤
        children_left = Int64.(a[arvore][1]) #nós a esquerda
        children_right = Int64.(a[arvore][2]) #nós a direita
        gammas = a[arvore][7] #predict encontrado no treino
        for k = 1:rows
            onde = 1 #posição do dado na árvore para saber em qual nó estamos
            while (isnt_leave[onde]==true) # verifica se o nó onde o dado está é folha ou nó
                if X[k,feature[onde]+1] ≤ threshold[onde] # condição do nó, if true -> foi para esquerda
                    onde = children_left[onde]+1
                else #caso false -> foi para nó direito
                    onde = children_right[onde]+1
                end
            end
            #println("----")
            #if contador%1 == 0
                #if length(resultado) !=0
                    #println("quase resposta:", sum(resultado)+f₀)
                #end
            #end
            #println(gammas[onde])
            #println("γ:",gammas[onde])

            resposta_arvore[k] = gammas[onde]
            #push!(resultado,gammas[onde]) # armazena os resultados das etapas
            #if k <= 3 && arvore <= 2
            #    println("----------")
            #    println("k:",k)
            #    println("arvore:",arvore)
            #    println(resposta[1:rows,tamanho])
            #    push!(R,sum(resultado)+f₀)
            #end
        end
        resposta[1:end,arvore+1] = resposta_arvore
    end

    #for k = 1:rows # para todos os dados do conjunto X
    #    resultado = [] #armazeno as respostas de todas as árvores para aquele dado

    #    f_pred[k] += sum(resultado)
    #end
    return sum(resposta,dims=2)#, r2(y_real,sum(resposta,dims=2))
end


function Fₘ_arvores(a, X, f₀, y_real; mom = 0, learning_rate = 0.01)
    if mom == 1
        learning_rate = 1.
    end
    tamanho = size(a)[1]
    rows = size(X)[1]
    f_pred = ones(rows)*f₀
    contador = 0
    i = 1
    Resultados_parciais = zeros(tamanho+1,rows)
    Resultados_parciais[1,1:end] .= f₀
    aux = [] #auxilia na construção da matriz R2
    for k = 1:rows
        resultado = []
        for arvore = 1:tamanho
            #println("arvore",arvore)
            isnt_leave = Int64.(a[arvore][6]) # é folha?
            feature = Int64.(a[arvore][4]) #coluna do conjunto utilizada naquele nó
            threshold = a[arvore][3] #valor de comparação do ≤
            children_left = Int64.(a[arvore][1]) #nós a esquerda
            children_right = Int64.(a[arvore][2]) #nós a direita
            gammas = a[arvore][7] #predict encontrado no treino
            onde = 1
            while (isnt_leave[onde]==true) #&& (t1-t2)/1e9 < 10.
                if X[k,feature[onde]+1] ≤ threshold[onde]
                    onde = children_left[onde]+1
                else
                    onde = children_right[onde]+1
                end
            end
            push!(resultado,gammas[onde]*learning_rate)
            contador += 1
        end
        Resultados_parciais[2:end,i] = resultado[:]
        f_pred[i] += sum(resultado)
        i += 1
    end
    R2 = zeros(tamanho+1)
    for k = 1:tamanho+1
        #println(size(sum(Resultados_parciais[1:k,1:end], dims = 1)'))
        #println(size(y_real))
        R2[k] = r2(y_real,sum(Resultados_parciais[1:k,1:end], dims = 1)')
    end
    return R2
end
