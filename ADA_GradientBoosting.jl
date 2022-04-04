function GB_Ada(X::Matrix,y::Array{Float64,1},
        learning_rate::Float64,n_estimators::Int64, μ::Float64;
        loss = "EQM",max_leaves = 5, n_iter_no_change = 0, tol = 1e-4, ϵ = 1e-8)
    #global a
    #func = Dict("EQM" => (0.5*(x[1] .- x[2]).^2),"DEQM" =>  (x[2].-x[1]))
    #deri = Dict("EQM" =>  (x[2].-x[1]))
    #Dict("A"=>1, "B"=>2).
    # x[1] = y, x[2] = imagem_f (y^)
    perda(x) = 0.5*(x[1] .- x[2]).^2
    derivada(x) = x[1].-x[2] # - derivada direto
    f₀ = mean(y)*ones(length(y))
    sᵢ = zeros(length(y))
    imagem_f = f₀[:]
    a = []
    #j=1
    #n_iter_no_change #serve para fazer uma parada
    for n = 1:n_estimators #adicionar critério de tempo
        #println("V",vᵢ)
        rᵢ = derivada([y,imagem_f]) # - derivada
        sᵢ = sᵢ + rᵢ.^2 #passo do momentum

        estimator = tree.DecisionTreeRegressor(
            max_leaf_nodes=max_leaves,random_state = 0) # weak learner
        vᵢ = learning_rate*rᵢ./sqrt.(sᵢ .+ ϵ) #mᵢ
        estimator.fit(X, vᵢ') #treino do weak learner
        #if n == 2
        #    println(vᵢ)
        #end
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
            γⱼₘ = sum(vᵢₘ)/Rⱼₘ
            #γⱼₘ = Optim.minimizer(optimize(γ -> sum(perda([yᵢ,imagem_f[indices].+γ])),[0.5], Newton()))[1] #ou Newton
            imagem_f[indices] .+= γⱼₘ
            gammas[foia] = γⱼₘ
            #if j<5
            #    println(gammas)
            #end
            #if any(x->abs(x)>= 1e5, imagem_f)
            #    return "Erro"
            #end
        end
        #println(gammas)
        if n ==1
            a = hcat(children_left, children_right, threshold, feature, val_folha[:],children_right-children_left, gammas)
        else
            a1 = hcat(children_left, children_right, threshold, feature, val_folha[:],children_right-children_left, gammas)
            a = cat(a,a1, dims = 3)
        end
        if n_iter_no_change != 0
            if tol >= norm(rᵢ)
                n_iter_no_change+=1
            end
        end
    end
    return [a, mean(y)]
end
