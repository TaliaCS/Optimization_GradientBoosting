function GB_nesterov(X::Matrix,y::Array{Float64,1},
        learning_rate::Float64,n_estimators::Int64, μ::Float64;
        loss = "EQM",max_leaves = 5, n_iter_no_change = 0, tol = 1e-4)
    #global a
    #func = Dict("EQM" => (0.5*(x[1] .- x[2]).^2),"DEQM" =>  (x[2].-x[1]))
    #deri = Dict("EQM" =>  (x[2].-x[1]))
    #Dict("A"=>1, "B"=>2).
    # x[1] = y, x[2] = imagem_f (y^)
    perda(x) = 0.5*(x[1] .- x[2]).^2
    derivada(x) = x[1].-x[2] # - derivada direto
    f₀ = mean(y)*ones(length(y))
    vᵢ = zeros(length(y))
    imagem_f = f₀[:]
    a = Any[]
    #j=1
    #n_iter_no_change #serve para fazer uma parada
    for n = 1:n_estimators #adicionar critério de tempo

        rᵢ = derivada([y,imagem_f + μ*vᵢ]) # - derivada

        vᵢ = μ*vᵢ + learning_rate*rᵢ #passo do momentum
        estimator = tree.DecisionTreeRegressor(
            max_leaf_nodes=max_leaves,random_state = 0) # weak learner

        estimator.fit(X, vᵢ') #treino do weak learner
        y_pred = estimator.predict(X) #predição do weak learner

        n_nodes = estimator.tree_.node_count #número de nós
        children_left = estimator.tree_.children_left #filhos a esquerda
        children_right = estimator.tree_.children_right #filhos a direita
        feature = estimator.tree_.feature #coluna usada por decisão
        threshold = estimator.tree_.threshold # valor da comparação
        val_folha = estimator.tree_.value # valor das folhas

        id_folhas = findall(x->x==1,children_right .== children_left) #id das regiões
        gammas = ones(n_nodes)*-1
        Rⱼₘ = length(id_folhas) #tamanho das regiões
        for foia = id_folhas
            indices = findall(x->x==1,val_folha[foia] .== y_pred)
            vᵢₘ = vᵢ[indices]
            γⱼₘ = sum(vᵢₘ)/length(vᵢₘ)
            imagem_f[indices] .+= γⱼₘ
            gammas[foia] = γⱼₘ
            #if j<5
            #    println(gammas)
            #end

        end
        #println(gammas)
        push!(a,[children_left, children_right, threshold, feature, val_folha[:],children_right-children_left, gammas])
        #if n ==1
        #    a = hcat(children_left, children_right, threshold, feature, val_folha[:],children_right-children_left, gammas)
        #else
        #    a1 = hcat(children_left, children_right, threshold, feature, val_folha[:],children_right-children_left, gammas)
        #    a = cat(a,a1, dims = 3)
        #end
        if n_iter_no_change != 0
            if tol >= norm(rᵢ)
                n_iter_no_change+=1
            end
        end
    end
    return (a, mean(y), sum(perda([y,imagem_f])))
end

#%%
k = 9
μ = 0.45
h_gb_0,f₀_gb_0 = GB_nesterov(X_train[k], y_train[k],
        0.0001, 100,μ, n_iter_no_change = 5, max_leaves = 4)
gb_test_0 = Fₘ_momentum(h_gb_0,X_test[k], f₀_gb_0)
aval(gb_test_0,y_test[k])
