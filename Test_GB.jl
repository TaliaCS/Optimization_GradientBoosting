LR = 0.01
n_arvores = 100
X_train = Any[[X_train_1]; [X_train_2] ;[X_train_3] ;[X_train_4] ;[X_train_5] ;[X_train_6] ;[X_train_7]; [X_train_8]; [X_train_9]]
y_train = Any[[y_train_1]; [y_train_2] ;[y_train_3] ;[y_train_4] ;[y_train_5] ;[y_train_6] ;[y_train_7]; [y_train_8]; [y_train_9]]
X_test = Any[[X_test_1]; [X_test_2]; [X_test_3]; [X_test_4] ;[X_test_5] ;[X_test_6] ;[X_test_7] ;[X_test_8] ;[X_test_9]]
y_test = Any[[y_test_1]; [y_test_2]; [y_test_3]; [y_test_4]; [y_test_5] ;[y_test_6] ;[y_test_7] ;[y_test_8] ;[y_test_9]]

for k = 1:9
    if (k ==2)||(k ==5)
        LR = 0.001
    elseif k == 9
        LR = 0.0001
    else
        LR = 0.01
    end
        println("Conjunto de Dados",k)
        n_arvores = 100
        if k == 7
            n_arvores = 70
        end
        h_gb_1,f₀_gb_1 = GradientBoostingRegressor(X_train[k], y_train[k],
            LR, n_arvores, n_iter_no_change = 5, max_leaves = 4)

        h_mom_1,f₀_mom_1 = GB_momentum(X_train[k], y_train[k],
            LR, n_arvores, 0.45, n_iter_no_change = 5, max_leaves = 4)

        h_nest_1,f₀_nest_1 = GB_nesterov(X_train[k], y_train[k],
            LR, n_arvores, 0.45, n_iter_no_change = 5, max_leaves = 4)

        h_ada_1,f₀_ada_1 = GB_Ada(X_train[k], y_train[k],
            LR, n_arvores, 0.45, n_iter_no_change = 5, max_leaves = 4,
            ϵ = 1e-8)

        #nest_test_1 = Fₘ_momentum(h_nest_1,X_test[k], f₀_nest_1)
        #gb_test_1 = Fₘ(h_gb_1,LR,X_test[k], f₀_gb_1) #predict nosso
        #mom_test_1 = Fₘ_momentum(h_mom_1,X_test[k], f₀_mom_1)
        r2_nes = Fₘ_arvores(h_nest_1,X_test[k], f₀_nest_1,y_test[k], mom = 1)
        r2_gb = Fₘ_arvores(h_gb_1,learning_rate = LR,X_test[k], f₀_gb_1, y_test[k])
        r2_mom = Fₘ_arvores(h_mom_1,X_test[k], f₀_mom_1,y_test[k], mom = 1)
        r2_ada = Fₘ_arvores(h_ada_1,X_test[k], f₀_ada_1,y_test[k], mom = 1)

        fig = plot(1:n_arvores+1, [r2_nes r2_gb r2_mom r2_ada], lw = 2, legend=:bottomright,
            label=["Nesterov" "GB" "Momentum" "Ada"], title = string("Conjunto de Dados ",k))
        xlabel!("Núm árvores")
        ylabel!("R²")
        display(fig)
        #savefig(fig, string("conjunto_r2_",k,".svg"))
        #println("Nosso GB")
        #aval(y_test[k],gb_test_1)
        #println("Momentum")
        #aval(y_test[k],mom_test_1)
        #println("Noterov")
        #aval(y_test[k],nest_test_1)
        #println("----------")
end


X_train = Any[[X_train_1] ;[X_train_2] ;[X_train_3] ;[X_train_4] ;[X_train_5] ;[X_train_6] ;[X_train_7]; [X_train_8]; [X_train_9]]
y_train = Any[[y_train_1] ;[y_train_2] ;[y_train_3] ;[y_train_4] ;[y_train_5] ;[y_train_6] ;[y_train_7]; [y_train_8]; [y_train_9]]
X_test = Any[[X_test_1]; [X_test_2]; [X_test_3]; [X_test_4] ;[X_test_5] ;[X_test_6] ;[X_test_7] ;[X_test_8] ;[X_test_9]]
y_test = Any[[y_test_1]; [y_test_2]; [y_test_3]; [y_test_4]; [y_test_5] ;[y_test_6] ;[y_test_7] ;[y_test_8] ;[y_test_9]]

for k = collect(Iterators.flatten((1:1,3:8)))
        fig = plot()
        println("Conjunto de Dados",k)
        if k >= 3
            LRₘ, μₘ, n_arvₘ = best_coef[k-1,1:3,1]
            LRₙ, μₙ, n_arvₙ = best_coef[k-1,1:3,2]
            LR , μ, n_arv = best_coef[k-1,1:3,3]
        else
            LRₘ, μₘ, n_arvₘ = best_coef[k,1:3,1]
            LRₙ, μₙ, n_arvₙ = best_coef[k,1:3,2]
            LR , μ, n_arv = best_coef[k,1:3,3]
        end
        n_arvₘ, n_arvₙ, n_arv = Int.((n_arvₘ, n_arvₙ, n_arv))

        h_gb_1,f₀_gb_1 = GradientBoostingRegressor(X_train[k], y_train[k],
            LR, n_arv, n_iter_no_change = 5, max_leaves = 4)

        h_mom_1,f₀_mom_1 = GB_momentum(X_train[k], y_train[k],
            LRₘ, n_arvₘ, μₘ, n_iter_no_change = 5, max_leaves = 4)

        h_nest_1,f₀_nest_1 = GB_nesterov(X_train[k], y_train[k],
            LRₙ, n_arvₙ, μₙ, n_iter_no_change = 5, max_leaves = 4)


        r2_nes = Fₘ_arvores(h_nest_1,X_test[k], f₀_nest_1,y_test[k], mom = 1)
        r2_gb = Fₘ_arvores(h_gb_1,learning_rate = LR,X_test[k], f₀_gb_1, y_test[k])
        r2_mom = Fₘ_arvores(h_mom_1,X_test[k], f₀_mom_1,y_test[k], mom = 1)

        aux = [(n_arvₘ, r2_mom, "Momentum", "Green");
               (n_arvₙ, r2_nes, "Nesterov", "Blue");
               (n_arv, r2_gb,  "GB", "Red")]
        p = sort(aux,  by = x -> x[1], rev=true)

        for i = 1:3
            if i == 1
                fig = plot(1:p[i][1]+1, p[i][2] , lw = 2, legend=:bottomright,
                    label= p[i][3], title = string("Conjunto de Dados ",k), linecolor = p[i][4])
            else
                fig = plot!(1:p[i][1]+1, p[i][2] , lw = 2, legend=:bottomright,
                    label= p[i][3], title = string("Conjunto de Dados ",k), linecolor = p[i][4])
            end
            xlabel!("Núm. árvores")
            ylabel!("R²")
        end
        display(fig)
        #savefig(fig, string("conjunto_r2_",k,".svg"))
        #println("Nosso GB")
        #aval(y_test[k],gb_test_1)
        #println("Momentum")
        #aval(y_test[k],mom_test_1)
        #println("Noterov")
        #aval(y_test[k],nest_test_1)
        #println("----------")
end
