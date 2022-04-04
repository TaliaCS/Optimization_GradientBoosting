function GradientBoostingRegressor(X::Matrix,y::Array{Float64,1},learning_rate::Float64,n_estimators::Int64;
        loss = "EQM",max_leaves = 5, n_iter_no_change = 0, tol = 1e-4)
    #global a
    #func = Dict("EQM" => (0.5*(x[1] .- x[2]).^2),"DEQM" =>  (x[2].-x[1]))
    #deri = Dict("EQM" =>  (x[2].-x[1]))
    #Dict("A"=>1, "B"=>2).
    # x[1] = y, x[2] = imagem_f (y^)
    perda(x) = 0.5*(x[1] .- x[2]).^2
    derivada(x) = x[1].-x[2] # - derivada direto
    f₀ = mean(y)*ones(length(y))
    imagem_f = f₀[:]
    a = Any[]
    #j=1
    #n_iter_no_change #serve para fazer uma parada
    for n = 1:n_estimators #adicionar critério de tempo

        rᵢ = derivada([y,imagem_f])
        #println("ri: ",norm(rᵢ))
        estimator = tree.DecisionTreeRegressor(max_leaf_nodes=max_leaves,
            random_state = 0)

        estimator.fit(X, rᵢ')
        #joblib.dump(estimator, "$(n).joblib")
        #estimator = joblib.load("$(k).joblib")
        y_pred = estimator.predict(X)

        n_nodes = estimator.tree_.node_count
        children_left = estimator.tree_.children_left
        children_right = estimator.tree_.children_right
        feature = estimator.tree_.feature
        threshold = estimator.tree_.threshold
        val_folha = estimator.tree_.value

        id_folhas = findall(x->x==1,children_right .== children_left)
        gammas = ones(n_nodes)*-1
        for foia = id_folhas
            indices = findall(x->x==1,val_folha[foia] .== y_pred)
            yᵢ = y[indices]

            y_predᵢ = imagem_f[indices]
            #γⱼₘ = sum((yᵢ - y_predᵢ ))/length(y_predᵢ)
            γⱼₘ = Optim.minimizer(optimize(γ -> sum(perda([yᵢ,imagem_f[indices].+γ])),[0.5], Newton()))[1] #ou Newton
            imagem_f[indices] .+= γⱼₘ*learning_rate
            gammas[foia] = γⱼₘ
            #if j<5
            #    println(gammas)
            #end

        end
        push!(a,[children_left, children_right, threshold,
                feature, val_folha[:],children_right-children_left,
                gammas])
        #println(gammas)

    end
    return (a, mean(y), sum(perda([y,imagem_f])))
end

#%%

k = 9
μ = 0.45
h_gb_0,f₀_gb_0 = GradientBoostingRegressor(X_train[k], y_train[k],
        0.1, 100, n_iter_no_change = 5, max_leaves = 4)
gb_test_0 = Fₘ_n(h_gb_0,0.1,X_test[k], f₀_gb_0)
aval(gb_test_0,y_test[k])
