X = [X_1, X_2, X_3, X_4, X_5, X_6, X_7, X_8 ,X_9]
y = [y_1, y_2, y_3, y_4, y_5, y_6, y_7, y_8 ,y_9]

function K_fold(modelo,X,y,k_fold,
                Learning_rate,n_arvores; μ = 0, mom = 1)

    tam_x = size(X)[1]
    aux = collect(range(1,length = k_fold+1,tam_x)) #pega o num de dados em X e divide em k_folds
    aux = Int64.(floor.(aux)) #caso a divisão não seja exata -> float -> int
    folds = Any[]
    pred_vet = Any[]
    index_data = shuffle(collect(1:tam_x)) # index das linhas dos dados bagunçados
    #MAE = 0.
    R2 = zeros(k_fold,n_arvores+1)
    for k = 2:k_fold+1
        index_aux = collect(Iterators.flatten((index_data[aux[1]:aux[k-1]],index_data[aux[k]:tam_x])))
        X_train = X[index_aux, 1:end]
        X_test = X[index_data[aux[k-1]:aux[k]], 1:end]
        y_train = y[index_aux]
        y_test = y[index_data[aux[k-1]:aux[k]]]
        if mom !=1
            h,f₀ =  modelo(X_train, y_train,Learning_rate, #treinando com o número maximo de arvores
            n_arvores[end],n_iter_no_change = 5, max_leaves = 4)
        else
            h,f₀ =  modelo(X_train, y_train,Learning_rate, #treinando com o número maximo de arvores
                n_arvores[end], μ, n_iter_no_change = 5, max_leaves = 4)
        end

        R2[k-1, 1:end] = Fₘ_arvores(h,X_test, f₀, y_test, mom = mom, learning_rate = Learning_rate )

        #erro_abs = abs.(y_test - pred)
        #MAE += sum(erro_abs)/length(erro_abs)
        #R2[k] = r2(y_test, pred_vet[k])
        #println("-------------")
        #println("Fold ", k-1)
        #println("MAE: ",sum(erro_abs)/length(erro_abs))
        #println("R2: ",r2(y_test, pred))
    end
    R2 = sum(R2, dims = 1)/k_fold
    posic_best = findall(x->x==maximum(R2), R2)
    #println("-------------")
    #println("Média dos resultados:")
    #println("MAE: ", MAE/k_fold)
    #println("R2: ", R2/k_fold)

    return [R2[posic_best][1],posic_best[1][2] - 1] #R2/k_fold
end

function GridSearch(modelo,X,y,Learning_rate,n_arvores; μ = 0, n_folds = 3, mom = 1)

    LR = collect(Learning_rate)
    μ = collect(μ)
    #n_arv = collect(n_arvores)
    combinacao = collect(Iterators.product(LR,μ))
    n_combinacao = length(combinacao)
    #MAE = zeros(n_combinacao)
    R2 = zeros(n_combinacao, 2)
    for k = 1:n_combinacao
        #println("iteração ",k,"  de ",n_combinacao)
        LR_k,μ_k = combinacao[k]
        R2[k,1:2] = K_fold(modelo,X,y,n_folds, LR_k, n_arvores, μ = μ_k, mom = mom)
        #erro_abs = abs.(y_test - pred)
        #MAE[k] = sum(erro_abs)/length(erro_abs)
    end
    posic_best = findall(x->x==maximum(R2[1:end,1]), R2[1:end,1])

    return R2[posic_best,1],[combinacao[posic_best], R2[posic_best,2]]
end

LR_mom_nest = 0.001:0.02:0.1
LR_gb = 0.1:0.1:0.9
n_arvores = 100
μ = 0.1:0.05:0.5

R = Any[]
push!(R, ["Momentum", "Nesterov", "Gradiente Boosting"])
println(["Momentum", "Nesterov", "Gradiente Boosting"])
Best_coef = Any[]
push!(Best_coef, ["Momentum", "Nesterov", "Gradiente Boosting"])
for k = collect(Iterators.flatten((2:2,9:9)))
    println("Conjunto de dados: ", k)
    LR_mom_nest = 0.85.^collect(1:4:41)#0.0001:0.02:0.09
    LR_gb = 0.1:0.1:0.9
    n_arvores = 100
    μ = 0.85.^collect(1:4:41)#0.1:0.05:0.5 #pego do artigo
    #println("GS Momentum")
    mom,b_mom = GridSearch(GB_momentum,X_train[k],y_train[k],LR_mom_nest,n_arvores, μ = μ )
    #println("GS Nesterov")
    nest,b_nest = GridSearch(GB_nesterov,X_train[k],y_train[k],LR_mom_nest,n_arvores, μ = μ )
    #println("GS Gradient")
    gb,b_gb = GridSearch(GradientBoostingRegressor,X_train[k],y_train[k],LR_gb,n_arvores, mom = 0)

    push!(R,[mom,nest,gb])
    push!(Best_coef,[b_mom,b_nest, b_gb])
    println(" R2: ", [mom,nest,gb])
    println("arv: ", [b_mom[2],b_nest[2], b_gb[2]])
end

best_coef = zeros(2,3,3)
for k = 2:3
    best_coef[k-1, 1:end, 1] .= Best_coef[k][1][1][1][1], Best_coef[k][1][1][1][2], Best_coef[k][1][2][1]
    best_coef[k-1, 1:end, 2] .= Best_coef[k][2][1][1][1], Best_coef[k][2][1][1][2], Best_coef[k][2][2][1]
    best_coef[k-1, 1:end, 3] .= Best_coef[k][3][1][1][1], Best_coef[k][3][1][1][2], Best_coef[k][3][2][1]
end
