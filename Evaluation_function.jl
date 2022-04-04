function aval(y_test, y_pred)
    ç1 = abs.(y_test - y_pred)
    #println("norma:",norm(ç1))
    #println("min:",minimum(ç1))
    #println("max:",maximum(ç1))
    #println("---")
    println("MAE:", sum(ç1)/length(ç1))
    println("R2:",r2(y_test, y_pred))
end
